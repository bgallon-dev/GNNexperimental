r"""v3.1 Phase 5 — manifold-index export.

Encodes a corpus split once with a trained v3.1 encoder and writes a
reusable artifact (NOT just a checkpoint): per-node Poincaré embedding
+ the metadata the retrieval ops and the future graph kernel need.

Per node row:
  graph_idx, node_idx, optional neo4j_node_id, embedding(D),
  radius (||logmap0(emb)||), out_degree, in_degree, collapse_flag
  (near-duplicate, same tau logic as eval_retrieval_midpoint:232-236),
  node_type, layer, depth.

Outputs:
  manifold_index.npz       — arrays (rows aligned across all fields)
  manifold_index_meta.json — config, encoder/query SHA, manifest ref,
                              corpus, split, tau_frac, depth_divisor.

Asserts the encoder SHA against the locked baseline manifest when
``--assert-sha --baseline-dir`` are given (proves the artifact indexes
the asset the manifest describes).

Usage
-----
    py -m src.modelsv3.export_manifold_index \
        --run runs/v3.1_qh2_infonce_seed1 --split val \
        --out runs/v3.1_qh2_infonce_seed1/manifold_index.npz \
        --assert-sha --baseline-dir runs/v3.1-baseline-hyp-h128-l4-seed1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv2.layers import poincare_ops as P  # noqa: E402
from src.modelsv3.eval_candidate_recall import (  # noqa: E402
    _build_encoder,
    build_query_encoder,
)
from src.modelsv3.eval_retrieval_nn import (  # noqa: E402
    _pairwise_distance_matrix,
    _unique_graph_indices,
)
from src.modelsv3.lock_baseline import sha256_file  # noqa: E402

DEPTH_DIVISOR = 5
DEFAULT_TAU_FRAC = 1e-4


def _collapse_flags(emb: torch.Tensor, c, euclidean: bool,
                    tau_frac: float) -> np.ndarray:
    """Flag the *redundant* member of each near-duplicate cluster, NOT
    every node in it (plan §7: collapse near-duplicates to one
    representative). A node is flagged iff it has a near-duplicate
    (dist < tau = tau_frac x median) at a strictly LOWER row index — so
    the first member of every collapse cluster stays unflagged and
    survives retrieval; the extra copies are flagged for de-duplication.
    tau convention matches eval_retrieval_midpoint."""
    D = _pairwise_distance_matrix(emb, c, euclidean).detach().cpu().numpy()
    np.fill_diagonal(D, np.inf)
    finite = D[np.isfinite(D)]
    N = emb.size(0)
    if finite.size == 0:
        return np.zeros(N, dtype=bool)
    tau = tau_frac * float(np.median(finite))
    flags = np.zeros(N, dtype=bool)
    for i in range(1, N):
        if (D[i, :i] < tau).any():
            flags[i] = True
    return flags


def export(run_dir: Path, split: str, out_path: Path, corpus: str,
           split_seed: int, task: int | None, tau_frac: float,
           assert_sha: bool, baseline_dir: Path | None) -> dict:
    run_dir = Path(run_dir)
    summary = run_dir / "summary.json"
    enc_path = run_dir / "encoder.pt"
    q_path = run_dir / "query_encoder.pt"
    cfg = json.loads(summary.read_text())["config"]

    enc_sha = sha256_file(enc_path)
    q_sha = sha256_file(q_path)
    if assert_sha and baseline_dir is not None:
        from src.modelsv3.lock_baseline import load_manifest
        man = load_manifest(baseline_dir)
        # The v3.1 encoder is the *trained* head's encoder; only assert
        # if this run IS the locked baseline. Otherwise just record both.
        if enc_sha == man["encoder_sha256"]:
            print("[export] encoder SHA matches the locked baseline.")
        else:
            print("[export] note: encoder SHA differs from locked baseline "
                  "(expected for a Phase-2/3 trained head); recorded in meta.")

    include_tasks = {task} if task is not None else None
    dataset = CorpusDataset(
        corpus_dir=corpus, split=split, split_seed=split_seed,
        include_tasks=include_tasks,
    )
    encoder = _build_encoder(cfg, dataset)
    encoder.load_state_dict(torch.load(enc_path, map_location="cpu"))
    encoder.eval()
    qenc = build_query_encoder(cfg, dataset)
    qenc.load_state_dict(torch.load(q_path, map_location="cpu"))
    qenc.eval()
    euclidean = cfg["model"] == "euclidean"
    c_val = getattr(encoder, "c", torch.tensor(float(cfg.get("curvature", 1.0))))

    graph_ids = _unique_graph_indices(dataset)
    cols: dict[str, list] = {
        "graph_idx": [], "node_idx": [], "neo4j_node_id": [],
        "radius": [], "out_degree": [],
        "in_degree": [], "collapse_flag": [], "node_type": [], "layer": [],
        "depth": [],
    }
    emb_rows: list[np.ndarray] = []
    with torch.no_grad():
        for gi in graph_ids:
            g = dataset._get_graph(gi)
            with np.load(dataset.files[gi]) as npz:
                neo4j_ids = (
                    npz["neo4j_node_id"].astype(np.int64)
                    if "neo4j_node_id" in npz.files else None
                )
            out = encoder(g["x"], g["edge_index"], g["edge_type"],
                          g["edge_descriptor"],
                          node_descriptor=g["node_descriptor"])
            emb = out.node_embeddings.detach()
            N = emb.size(0)
            ei = g["edge_index"].cpu().numpy()
            outdeg = np.bincount(ei[0], minlength=N)
            indeg = np.bincount(ei[1], minlength=N)
            radius = (
                emb.norm(dim=-1).cpu().numpy() if euclidean
                else P.logmap0(emb, c_val).norm(dim=-1).cpu().numpy()
            )
            cflags = _collapse_flags(emb, c_val, euclidean, tau_frac)
            x = g["x"].cpu().numpy()
            ntype = np.where(x[:, 0:12].sum(1) > 0, x[:, 0:12].argmax(1), -1)
            layer = np.where(x[:, 12:16].sum(1) > 0, x[:, 12:16].argmax(1), -1)
            depth = np.rint(x[:, 20] * DEPTH_DIVISOR).astype(np.int64)

            emb_rows.append(emb.cpu().numpy())
            for i in range(N):
                cols["graph_idx"].append(int(gi))
                cols["node_idx"].append(i)
                cols["neo4j_node_id"].append(
                    int(neo4j_ids[i]) if neo4j_ids is not None else -1)
                cols["radius"].append(float(radius[i]))
                cols["out_degree"].append(int(outdeg[i]))
                cols["in_degree"].append(int(indeg[i]))
                cols["collapse_flag"].append(bool(cflags[i]))
                cols["node_type"].append(int(ntype[i]))
                cols["layer"].append(int(layer[i]))
                cols["depth"].append(int(depth[i]))

    embeddings = np.concatenate(emb_rows, axis=0).astype(np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embedding=embeddings,
        graph_idx=np.array(cols["graph_idx"], dtype=np.int64),
        node_idx=np.array(cols["node_idx"], dtype=np.int64),
        neo4j_node_id=np.array(cols["neo4j_node_id"], dtype=np.int64),
        radius=np.array(cols["radius"], dtype=np.float32),
        out_degree=np.array(cols["out_degree"], dtype=np.int64),
        in_degree=np.array(cols["in_degree"], dtype=np.int64),
        collapse_flag=np.array(cols["collapse_flag"], dtype=bool),
        node_type=np.array(cols["node_type"], dtype=np.int64),
        layer=np.array(cols["layer"], dtype=np.int64),
        depth=np.array(cols["depth"], dtype=np.int64),
    )
    meta = {
        "index_version": "kgr-v3.1",
        "run_dir": str(run_dir),
        "corpus": corpus,
        "split": split,
        "split_seed": split_seed,
        "task": task,
        "n_nodes": int(embeddings.shape[0]),
        "n_graphs": len(graph_ids),
        "dim": int(embeddings.shape[1]),
        "model": cfg["model"],
        "curvature": float(cfg.get("curvature", 1.0)),
        "query_head_arch": cfg.get("query_head_arch", "qh0"),
        "encoder_sha256": enc_sha,
        "query_encoder_sha256": q_sha,
        "baseline_manifest_ref": (
            str(Path(baseline_dir) / "baseline_manifest.json")
            if baseline_dir else None
        ),
        "tau_frac": tau_frac,
        "depth_divisor": DEPTH_DIVISOR,
        "collapse_frac": float(np.mean(cols["collapse_flag"])),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote {out_path}  ({meta['n_nodes']} nodes, "
          f"{meta['n_graphs']} graphs, dim={meta['dim']}, "
          f"collapse_frac={meta['collapse_frac']:.4f})")
    print(f"[export] meta: {meta_path}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str, required=True,
                    help="v3.1 run dir with encoder.pt/query_encoder.pt/"
                         "summary.json")
    ap.add_argument("--split", type=str, default="val",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--task", type=int, default=2, help="-1 for all tasks.")
    ap.add_argument("--tau-frac", type=float, default=DEFAULT_TAU_FRAC)
    ap.add_argument("--assert-sha", action="store_true")
    ap.add_argument("--baseline-dir", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    out = Path(args.out) if args.out else run_dir / "manifold_index.npz"
    export(
        run_dir=run_dir, split=args.split, out_path=out, corpus=args.corpus,
        split_seed=args.split_seed,
        task=None if args.task < 0 else int(args.task),
        tau_frac=args.tau_frac, assert_sha=args.assert_sha,
        baseline_dir=Path(args.baseline_dir) if args.baseline_dir else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
