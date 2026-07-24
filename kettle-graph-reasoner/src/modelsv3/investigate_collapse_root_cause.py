r"""Root-cause investigation for v3 embedding collapse.

Answers three diagnostic questions about the nodes that participate in
collapsed pairs (dist < 1e-4 × graph median dist):

    Q1 — Where on the manifold are collapsed nodes?  For hyperbolic,
         radius from ball origin. For Euclidean, L2 norm.  If
         collapsed nodes cluster at low radius/norm, it's an origin
         attractor.

    Q2 — What graph-structural and label properties do collapsed nodes
         share?  For each node-level feature (degree, clustering,
         depth, type, layer), compare the distribution among collapsed
         nodes vs non-collapsed nodes.  Cohen's d for continuous;
         proportion comparison for categorical.

    Q3 — Do collapsed nodes have similar input features (i.e. is the
         encoder faithfully representing similar inputs, or is it
         introducing collapse on dissimilar inputs)?  Full-vector
         input cosine similarity of collapsed pairs vs random pairs.

Usage
-----
    py src/modelsv3/investigate_collapse_root_cause.py \\
        --checkpoint runs/v3_hyp_compute_seed0/encoder.pt \\
        --out        runs/v3_hyp_compute_seed0/collapse_root_cause.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv2.layers import poincare_ops as P  # noqa: E402
from src.modelsv3.euclidean_v3 import EuclideanReasonerV3  # noqa: E402
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3  # noqa: E402


NODE_TYPE_SLICE = slice(0, 12)
LAYER_SLICE = slice(12, 16)
STRUCT_SLICE = slice(16, 21)
TEMPORAL_SLICE = slice(21, 24)
IDENTITY_SLICE = slice(24, 32)
STRUCT_FEATURE_NAMES = ("log_deg", "log_in", "log_out", "clustering", "depth_norm")
TEMPORAL_FEATURE_NAMES = ("t_start", "t_end", "duration")
DEFAULT_TAU_FRAC = 1e-4


# ---------------------------------------------------------------------------
# model loading (reused pattern)
# ---------------------------------------------------------------------------

def _load_encoder(
    checkpoint: Path, summary: Path, dataset: CorpusDataset
) -> tuple[torch.nn.Module, dict]:
    with open(summary, "r") as f:
        s = json.load(f)
    cfg = s["config"]
    if cfg["model"] == "hyperbolic":
        model = KettleGraphReasonerV3(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            type_dim=int(cfg["type_dim"]),
            c=float(cfg["curvature"]),
            num_edge_types_max=dataset.num_edge_types_max,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
            tangent_scale_init=float(cfg.get("tangent_scale", 0.1)),
        )
    elif cfg["model"] == "euclidean":
        model = EuclideanReasonerV3(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            type_dim=int(cfg["type_dim"]),
            num_edge_types_max=dataset.num_edge_types_max,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
        )
    else:
        raise ValueError(f"unknown model kind {cfg['model']!r}")
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def _unique_graph_indices(dataset: CorpusDataset) -> list[int]:
    seen: list[int] = []
    seen_set: set[int] = set()
    for graph_idx, _task_idx in dataset.index:
        gi = int(graph_idx)
        if gi not in seen_set:
            seen_set.add(gi)
            seen.append(gi)
    return seen


def _pairwise_dist(emb: Tensor, c: float | Tensor, euclidean: bool) -> Tensor:
    if euclidean:
        return torch.cdist(emb, emb, p=2)
    u = emb.unsqueeze(1)
    v = emb.unsqueeze(0)
    return P.dist(u, v, c, keepdim=False)


# ---------------------------------------------------------------------------
# core: identify collapsed pairs and the nodes that participate
# ---------------------------------------------------------------------------

def _collapsed_pairs_and_counts(
    D: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """For a given pairwise distance matrix D, return:
       pairs: (P, 2) array of (i, j) with i < j and dist < tau
       n_collapsed_per_node: (N,) — how many collapsed pairs each node is in"""
    N = D.shape[0]
    iu = np.triu_indices(N, k=1)
    dists = D[iu]
    finite = np.isfinite(dists)
    mask = finite & (dists < tau)
    pi = iu[0][mask]
    pj = iu[1][mask]
    pairs = np.stack([pi, pj], axis=1)
    counts = np.zeros(N, dtype=np.int32)
    for i in pi:
        counts[int(i)] += 1
    for j in pj:
        counts[int(j)] += 1
    return pairs, counts


# ---------------------------------------------------------------------------
# Q1: manifold location
# ---------------------------------------------------------------------------

def _q1_location(
    emb: Tensor,
    collapsed_node_mask: np.ndarray,
    c_val: float | Tensor,
    euclidean: bool,
) -> dict:
    """Compute embedding norm (L2 for Euclidean; ball-radius for
    hyperbolic).  For hyperbolic, ball radius is simply ||p||_2 where
    p is on the Poincaré ball and ||p|| < 1/sqrt(c).

    Compare distribution among collapsed vs non-collapsed nodes."""
    emb_np = emb.detach().cpu().numpy()
    norms = np.linalg.norm(emb_np, axis=1)

    coll = norms[collapsed_node_mask]
    ncoll = norms[~collapsed_node_mask]

    def describe(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {"n": 0}
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    collapsed = describe(coll)
    non_collapsed = describe(ncoll)

    # Cohen's d between the two distributions
    d = float("nan")
    if coll.size > 1 and ncoll.size > 1:
        pooled_var = (coll.var(ddof=1) * (coll.size - 1)
                      + ncoll.var(ddof=1) * (ncoll.size - 1)) / (coll.size + ncoll.size - 2)
        if pooled_var > 0:
            d = float((coll.mean() - ncoll.mean()) / math.sqrt(pooled_var))

    # For hyperbolic, compute what fraction of collapsed nodes sit within
    # epsilon of the origin. Meaningful only for hyperbolic.
    if not euclidean:
        eps_values = (1e-4, 1e-3, 1e-2, 1e-1)
        frac_near_origin = {
            f"r<{e}": float((coll < e).mean()) if coll.size > 0 else float("nan")
            for e in eps_values
        }
    else:
        frac_near_origin = None

    return {
        "metric": "ball_radius" if not euclidean else "L2_norm",
        "collapsed_nodes": collapsed,
        "non_collapsed_nodes": non_collapsed,
        "cohen_d_collapsed_vs_not": d,
        "fraction_near_origin": frac_near_origin,
    }


# ---------------------------------------------------------------------------
# Q2: graph-structural and label properties of collapsed nodes
# ---------------------------------------------------------------------------

def _q2_properties(
    x_np: np.ndarray,
    edge_index: np.ndarray,
    collapsed_node_mask: np.ndarray,
) -> dict:
    """For each structural feature, compare collapsed vs non-collapsed
    distributions. For categorical type/layer, compute proportion entropy.
    """
    N = x_np.shape[0]

    # Recover per-node structural features from x block [16:21].
    struct = x_np[:, STRUCT_SLICE]  # (N, 5)
    temporal = x_np[:, TEMPORAL_SLICE]
    types = np.where(
        x_np[:, NODE_TYPE_SLICE].sum(axis=1) > 0,
        x_np[:, NODE_TYPE_SLICE].argmax(axis=1), -1,
    )
    layers = np.where(
        x_np[:, LAYER_SLICE].sum(axis=1) > 0,
        x_np[:, LAYER_SLICE].argmax(axis=1), -1,
    )

    results: dict[str, dict] = {}

    # Continuous features: Cohen's d per feature
    for name_group, arr, names in (
        ("structural", struct, STRUCT_FEATURE_NAMES),
        ("temporal", temporal, TEMPORAL_FEATURE_NAMES),
    ):
        group: dict[str, dict] = {}
        for col, feat_name in enumerate(names):
            vals = arr[:, col]
            coll = vals[collapsed_node_mask]
            ncoll = vals[~collapsed_node_mask]
            if coll.size < 2 or ncoll.size < 2:
                group[feat_name] = {"cohen_d": float("nan"),
                                    "mean_collapsed": float(coll.mean()) if coll.size else float("nan"),
                                    "mean_non_collapsed": float(ncoll.mean()) if ncoll.size else float("nan")}
                continue
            pooled = (coll.var(ddof=1) * (coll.size - 1)
                      + ncoll.var(ddof=1) * (ncoll.size - 1)) / (coll.size + ncoll.size - 2)
            d = ((coll.mean() - ncoll.mean()) / math.sqrt(pooled)
                 if pooled > 0 else float("nan"))
            group[feat_name] = {
                "cohen_d": float(d),
                "mean_collapsed": float(coll.mean()),
                "mean_non_collapsed": float(ncoll.mean()),
                "std_collapsed": float(coll.std()),
                "std_non_collapsed": float(ncoll.std()),
            }
        results[name_group] = group

    # Categorical features: type and layer distributions
    def _dist(values: np.ndarray, universe_size: int) -> dict:
        valid = values[values >= 0]
        if valid.size == 0:
            return {"n": 0, "entropy": float("nan")}
        counts = np.bincount(valid, minlength=universe_size)
        props = counts / counts.sum()
        # Shannon entropy (natural log)
        nz = props[props > 0]
        H = float(-np.sum(nz * np.log(nz)))
        return {
            "n": int(valid.size),
            "proportions": [float(p) for p in props],
            "entropy": H,
            "max_entropy": float(math.log(universe_size)),
            "gini_diversity": float(1.0 - np.sum(props ** 2)),
        }

    results["type_distribution_collapsed"] = _dist(types[collapsed_node_mask], 12)
    results["type_distribution_non_collapsed"] = _dist(types[~collapsed_node_mask], 12)
    results["layer_distribution_collapsed"] = _dist(layers[collapsed_node_mask], 4)
    results["layer_distribution_non_collapsed"] = _dist(layers[~collapsed_node_mask], 4)

    return results


# ---------------------------------------------------------------------------
# Q3: input-feature similarity of collapsed pairs
# ---------------------------------------------------------------------------

def _q3_input_similarity(
    x_np: np.ndarray,
    collapsed_pairs: np.ndarray,
) -> dict:
    """Compare input-feature cosine similarity between collapsed pairs
    and random pairs.  Full-vector cosine plus per-block.

    If collapsed-pair cosine >> random-pair cosine, the encoder is
    faithfully mapping similar inputs to similar outputs — collapse is
    a data consequence, not an encoder pathology.
    """
    N = x_np.shape[0]
    iu = np.triu_indices(N, k=1)
    all_pairs = np.stack(iu, axis=1)

    # Sample a size-matched random control.  Avoid including collapsed
    # pairs in the random control by drawing from the full pool; with
    # N ≈ 200 and collapsed fraction ~10%, the overlap is small and
    # doesn't materially change the baseline.
    rng = np.random.default_rng(seed=1234)
    n_coll = len(collapsed_pairs)
    n_rand = min(max(n_coll * 2, 50), len(all_pairs))
    rand_indices = rng.choice(len(all_pairs), size=n_rand, replace=False)
    rand_pairs = all_pairs[rand_indices]

    def cosine(pairs: np.ndarray, feat_slice: slice | None) -> np.ndarray:
        if feat_slice is None:
            a = x_np[pairs[:, 0]]
            b = x_np[pairs[:, 1]]
        else:
            a = x_np[pairs[:, 0], feat_slice]
            b = x_np[pairs[:, 1], feat_slice]
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        denom = na * nb
        out = np.full(len(pairs), np.nan, dtype=np.float64)
        ok = denom > 0
        out[ok] = np.sum(a[ok] * b[ok], axis=1) / denom[ok]
        return out

    def describe(sims: np.ndarray) -> dict:
        clean = sims[~np.isnan(sims)]
        if clean.size == 0:
            return {"n": 0, "mean": float("nan"), "median": float("nan")}
        return {
            "n": int(clean.size),
            "mean": float(clean.mean()),
            "median": float(np.median(clean)),
            "std": float(clean.std()),
            "min": float(clean.min()),
            "max": float(clean.max()),
        }

    blocks = {
        "full_vector": None,
        "type": NODE_TYPE_SLICE,
        "layer": LAYER_SLICE,
        "structural": STRUCT_SLICE,
        "temporal": TEMPORAL_SLICE,
        "identity": IDENTITY_SLICE,
    }
    out: dict[str, dict | int] = {}
    for name, sl in blocks.items():
        coll_sims = cosine(collapsed_pairs, sl) if n_coll > 0 else np.array([])
        rand_sims = cosine(rand_pairs, sl)
        out[name] = {
            "collapsed_pairs": describe(coll_sims),
            "random_pairs": describe(rand_sims),
        }
    out["n_collapsed_pairs"] = int(n_coll)
    out["n_random_pairs"] = int(n_rand)
    return out


# ---------------------------------------------------------------------------
# top-level driver
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint: Path,
    summary: Path,
    corpus_dir: str,
    split: str,
    split_seed: int,
    task: int | None,
    tau_frac: float,
    out_path: Path,
) -> dict:
    include_tasks = {task} if task is not None else None
    dataset = CorpusDataset(
        corpus_dir=corpus_dir, split=split, split_seed=split_seed,
        include_tasks=include_tasks,
    )
    graph_ids = _unique_graph_indices(dataset)
    print(f"[root-cause] {len(dataset)} samples, {len(graph_ids)} graphs, tau={tau_frac}×med")

    model, cfg = _load_encoder(checkpoint, summary, dataset)
    euclidean = cfg["model"] == "euclidean"
    c_val = getattr(model, "c", torch.tensor(float(cfg["curvature"])))

    per_graph: list[dict] = []
    # Collapsed-node-index lists for cross-seed aggregation.
    per_graph_collapsed_nodes: dict[int, list[int]] = {}

    with torch.no_grad():
        for gi in graph_ids:
            graph = dataset._get_graph(gi)
            x = graph["x"]
            edge_index = graph["edge_index"]
            out = model(
                x, edge_index, graph["edge_type"], graph["edge_descriptor"],
                node_descriptor=graph["node_descriptor"],
            )
            emb = out.node_embeddings.detach().cpu()
            x_np = x.detach().cpu().numpy()
            ei_np = edge_index.detach().cpu().numpy()
            D = _pairwise_dist(emb, c_val, euclidean).detach().cpu().numpy()
            np.fill_diagonal(D, np.inf)
            finite = D[np.isfinite(D)]
            median_dist = float(np.median(finite)) if finite.size > 0 else float("nan")
            if not (median_dist == median_dist) or median_dist <= 0:
                continue
            tau = tau_frac * median_dist

            pairs, counts = _collapsed_pairs_and_counts(D, tau)
            N = x.size(0)
            collapsed_mask = counts > 0

            q1 = _q1_location(emb, collapsed_mask, c_val, euclidean)
            q2 = _q2_properties(x_np, ei_np, collapsed_mask)
            q3 = _q3_input_similarity(x_np, pairs)

            per_graph.append({
                "graph_idx": gi,
                "n_nodes": N,
                "n_collapsed_pairs": int(len(pairs)),
                "n_collapsed_nodes": int(collapsed_mask.sum()),
                "frac_collapsed_nodes": float(collapsed_mask.mean()),
                "median_dist": median_dist,
                "tau_absolute": tau,
                "q1": q1,
                "q2": q2,
                "q3": q3,
            })
            per_graph_collapsed_nodes[gi] = [
                int(i) for i, c in enumerate(counts) if c > 0
            ]

    # Aggregate across graphs.
    results = {
        "checkpoint": str(checkpoint),
        "model_kind": cfg["model"],
        "tau_frac": tau_frac,
        "n_graphs": len(per_graph),
        "per_graph": per_graph,
        "collapsed_node_ids_per_graph": per_graph_collapsed_nodes,
        "aggregate": _aggregate(per_graph, euclidean=euclidean),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)
    return results


def _aggregate(per_graph: list[dict], euclidean: bool) -> dict:
    """Summarise across graphs.  Key metrics:
       - Q1: Cohen's d of norm, collapsed vs non-collapsed
       - Q2: Cohen's d per structural feature; layer proportion entropy
       - Q3: full-vector cosine similarity, collapsed vs random
    """
    def mean_attr(path: list[str | int]) -> float:
        vals = []
        for g in per_graph:
            cur: object = g
            ok = True
            for p in path:
                if isinstance(p, str):
                    if not isinstance(cur, dict) or p not in cur:
                        ok = False
                        break
                    cur = cur[p]
                else:
                    if not isinstance(cur, list) or len(cur) <= p:
                        ok = False
                        break
                    cur = cur[p]
            if ok and isinstance(cur, (int, float)) and cur == cur and math.isfinite(cur):
                vals.append(float(cur))
        return statistics.mean(vals) if vals else float("nan")

    agg: dict = {
        "q1_norm_collapsed_vs_not_cohens_d": mean_attr(
            ["q1", "cohen_d_collapsed_vs_not"]
        ),
        "q1_norm_collapsed_mean": mean_attr(["q1", "collapsed_nodes", "mean"]),
        "q1_norm_non_collapsed_mean": mean_attr(["q1", "non_collapsed_nodes", "mean"]),
    }
    if not euclidean:
        near_origin_frac = []
        for g in per_graph:
            f = g["q1"].get("fraction_near_origin", {})
            if f and "r<0.001" in f and f["r<0.001"] == f["r<0.001"]:
                near_origin_frac.append(f["r<0.001"])
        agg["q1_frac_collapsed_within_r_1e-3"] = (
            statistics.mean(near_origin_frac) if near_origin_frac else float("nan")
        )

    # Q2: Cohen's d for structural and temporal features
    q2_cohens: dict[str, list[float]] = {}
    for g in per_graph:
        for block in ("structural", "temporal"):
            blk = g["q2"].get(block, {})
            for feat, info in blk.items():
                cd = info.get("cohen_d")
                if cd == cd and isinstance(cd, (int, float)) and math.isfinite(cd):
                    q2_cohens.setdefault(f"{block}.{feat}", []).append(float(cd))
    agg["q2_cohens_d_mean_across_graphs"] = {
        k: statistics.mean(v) for k, v in q2_cohens.items()
    }

    # Q2: per-layer proportions in collapsed vs non-collapsed — concentration
    coll_layer_entropy = [g["q2"]["layer_distribution_collapsed"]["entropy"]
                          for g in per_graph
                          if "entropy" in g["q2"]["layer_distribution_collapsed"]
                          and g["q2"]["layer_distribution_collapsed"]["entropy"]
                              == g["q2"]["layer_distribution_collapsed"]["entropy"]]
    ncoll_layer_entropy = [g["q2"]["layer_distribution_non_collapsed"]["entropy"]
                           for g in per_graph
                           if "entropy" in g["q2"]["layer_distribution_non_collapsed"]
                           and g["q2"]["layer_distribution_non_collapsed"]["entropy"]
                               == g["q2"]["layer_distribution_non_collapsed"]["entropy"]]
    agg["q2_layer_entropy_collapsed"] = (
        statistics.mean(coll_layer_entropy) if coll_layer_entropy else float("nan")
    )
    agg["q2_layer_entropy_non_collapsed"] = (
        statistics.mean(ncoll_layer_entropy) if ncoll_layer_entropy else float("nan")
    )

    coll_type_entropy = [g["q2"]["type_distribution_collapsed"]["entropy"]
                         for g in per_graph
                         if "entropy" in g["q2"]["type_distribution_collapsed"]
                         and g["q2"]["type_distribution_collapsed"]["entropy"]
                             == g["q2"]["type_distribution_collapsed"]["entropy"]]
    ncoll_type_entropy = [g["q2"]["type_distribution_non_collapsed"]["entropy"]
                          for g in per_graph
                          if "entropy" in g["q2"]["type_distribution_non_collapsed"]
                          and g["q2"]["type_distribution_non_collapsed"]["entropy"]
                              == g["q2"]["type_distribution_non_collapsed"]["entropy"]]
    agg["q2_type_entropy_collapsed"] = (
        statistics.mean(coll_type_entropy) if coll_type_entropy else float("nan")
    )
    agg["q2_type_entropy_non_collapsed"] = (
        statistics.mean(ncoll_type_entropy) if ncoll_type_entropy else float("nan")
    )

    # Q3: mean cosine of collapsed vs random for each feature block
    q3_means: dict[str, dict[str, list[float]]] = {}
    for g in per_graph:
        q3 = g["q3"]
        for block in ("full_vector", "type", "layer", "structural", "temporal", "identity"):
            if block in q3:
                q3_means.setdefault(block, {"collapsed": [], "random": []})
                cm = q3[block]["collapsed_pairs"].get("mean")
                rm = q3[block]["random_pairs"].get("mean")
                if cm == cm and isinstance(cm, (int, float)) and math.isfinite(cm):
                    q3_means[block]["collapsed"].append(cm)
                if rm == rm and isinstance(rm, (int, float)) and math.isfinite(rm):
                    q3_means[block]["random"].append(rm)
    agg["q3_cosine_similarity"] = {
        block: {
            "collapsed_mean": statistics.mean(d["collapsed"]) if d["collapsed"] else float("nan"),
            "random_mean": statistics.mean(d["random"]) if d["random"] else float("nan"),
        }
        for block, d in q3_means.items()
    }
    return agg


def _print_summary(r: dict) -> None:
    print()
    print("=" * 96)
    print(f"Root-cause investigation — {r['n_graphs']} graphs, "
          f"tau = {r['tau_frac']} × median_dist")
    print(f"checkpoint: {r['checkpoint']}   model: {r['model_kind']}")
    print("=" * 96)

    a = r["aggregate"]

    print("\nQ1 — Where on the manifold are collapsed nodes?")
    print(f"  norm (collapsed)        mean = {a['q1_norm_collapsed_mean']:.4f}")
    print(f"  norm (non-collapsed)    mean = {a['q1_norm_non_collapsed_mean']:.4f}")
    print(f"  Cohen's d               = {a['q1_norm_collapsed_vs_not_cohens_d']:+.3f}")
    if "q1_frac_collapsed_within_r_1e-3" in a:
        print(f"  fraction within r<1e-3  = {a['q1_frac_collapsed_within_r_1e-3']:.4f}")
    print("  Interpretation: large negative Cohen's d = collapsed nodes at smaller radii (origin attractor).")

    print("\nQ2 — Structural / temporal feature differences (Cohen's d; collapsed vs not)")
    for name, d in a["q2_cohens_d_mean_across_graphs"].items():
        print(f"  {name:<26}  d = {d:+.3f}")
    print(f"\n  Type entropy  collapsed={a['q2_type_entropy_collapsed']:.3f}  "
          f"non_collapsed={a['q2_type_entropy_non_collapsed']:.3f}")
    print(f"  Layer entropy collapsed={a['q2_layer_entropy_collapsed']:.3f}  "
          f"non_collapsed={a['q2_layer_entropy_non_collapsed']:.3f}")
    print("  Interpretation: lower entropy in collapsed = label concentration (cause B).")

    print("\nQ3 — Input cosine similarity: collapsed pairs vs random pairs")
    print(f"  {'block':<12}  {'collapsed':>14}  {'random':>14}  {'elevation':>12}")
    for block, d in a["q3_cosine_similarity"].items():
        cm = d["collapsed_mean"]
        rm = d["random_mean"]
        elev = (cm / rm) if (rm == rm and rm != 0) else float("nan")
        print(f"  {block:<12}  {cm:+14.4f}  {rm:+14.4f}  {elev:+12.2f}×")
    print("  Interpretation: high elevation = collapsed pairs have similar inputs (encoder faithful).")
    print("                   low or near-1× elevation = encoder is introducing collapse.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--summary", type=str, default=None)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--task", type=int, default=2)
    p.add_argument("--tau-frac", type=float, default=DEFAULT_TAU_FRAC)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    summary = Path(args.summary) if args.summary else checkpoint.parent / "summary.json"
    out = Path(args.out) if args.out else checkpoint.parent / "collapse_root_cause.json"
    evaluate_checkpoint(
        checkpoint=checkpoint, summary=summary, corpus_dir=args.corpus,
        split=args.split, split_seed=args.split_seed,
        task=None if args.task < 0 else args.task,
        tau_frac=args.tau_frac, out_path=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
