r"""KGR serving-layer verification harness.

Implements the gating checks from the plan (absolute numbers only; never
gap-closed fractions; honest negatives stated plainly):

  P0  schema-map no-op golden    -- the de-hardcoded SchemaMap reproduces
                                    the hardcoded neo4j_eval_export
                                    reference bit-for-bit (no DB needed).
  P1  bit-exact live parity      -- a live-pulled subgraph, run through
                                    the refactored tensor_contract + the
                                    SHA-asserted frozen encoder, produces
                                    per-node embeddings byte-identical
                                    (max abs diff < 1e-5) to the reference
                                    export pipeline (_encode_graph ->
                                    _build_graph_tensors -> encoder, via
                                    export_manifold_index) on the SAME
                                    node set. This is the spine: it proves
                                    the live path == the validated path.

P1 builds its own ground truth from the live DB (the reference corpus +
its manifold index do not ship; they are regenerated on demand so the
check is always against a current, honest reference). If Neo4j is
unreachable P1 reports BLOCKED (skipped) -- never a fake pass.

P2 (ndcg vs floor), P3 (locality control), P4 (scale/latency) are scoped
per the plan: the task-2 P2 path is already exercised end-to-end by
``scripts/smoke_retrieval_workflow.py`` + ``neo4j_eval_export compare``
(real-temporal, validated 0.558 -> 0.795); tasks 0/4/5 P2 is the
documented G1-gated follow-on (synthetic head zero-shot, honest negative
expected). This harness gates P0/P1; it points at P2-P4 rather than
re-deriving them.

Usage
-----
    py -m src.service.verify                 # P0 + P1 (P1 builds live ref)
    py -m src.service.verify --p0-only       # offline only
    py -m src.service.verify --ref-graphs 3 --ref-max-nodes 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.service.determinism import (  # noqa: E402
    child_env,
    ensure_pythonhashseed,
)
from src.service.schema_map import SchemaMap, golden_noop_check  # noqa: E402

_BASELINE = _ROOT / "runs" / "v3.1-baseline-hyp-h128-l4-seed1"
_KETTLE_CFG = _SCRIPTS / "kettle_config.yaml"
_P1_TOL = 1e-5


# ---------------------------------------------------------------------------
# P0
# ---------------------------------------------------------------------------

def run_p0() -> dict:
    sm = SchemaMap.from_yaml()
    cfg = yaml.safe_load(_KETTLE_CFG.read_text(encoding="utf-8")) or {}
    labels = sorted(set(cfg.get("required_properties", {}).keys())
                    | set(cfg.get("unique_keys", {}).keys()))
    rel_probe = [
        "SOURCED_FROM", "DERIVED_FROM", "PROVENANCE_OF", "EVIDENCED_BY",
        "REFERS_TO", "MENTIONS", "ABOUT", "DESCRIBES", "IN_YEAR",
        "HAS_PERIOD", "TEMPORAL_SCOPE", "DATED", "SUPERSEDES",
        "CORROBORATES", "SCOPED_TO", "CO_OCCURS_WITH", "RELATED_TO",
        "ASSOCIATED_WITH", "SAME_AS", "LINKED_TO", "NEAR", "OCCURRED_AT",
        "AT_PLACE", "HAS_HABITAT", "PART_OF", "OBSERVED_AT",
    ]
    rep = golden_noop_check(sm, labels, rel_probe)
    rep["gate"] = "P0"
    rep["pass"] = bool(rep["ok"])
    return rep


# ---------------------------------------------------------------------------
# P1
# ---------------------------------------------------------------------------

def _neo4j_reachable() -> bool:
    try:
        from neo4j_eval_export import _driver, _session  # type: ignore

        d = _driver()
        d.verify_connectivity()
        with _session(d) as s:
            s.run("RETURN 1").single()
        d.close()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [P1] Neo4j unreachable: {type(e).__name__}: {e}")
        return False


def _build_reference(ref_dir: Path, n_graphs: int, max_nodes: int,
                     seed: int) -> Path:
    """Regenerate the ground truth from the live DB: the reference
    exporter (_encode_graph, the thing we de-hardcoded) -> a tier1 NPZ
    corpus WITH neo4j ids, then export_manifold_index with the
    SHA-asserted frozen baseline encoder."""
    ref_dir.mkdir(parents=True, exist_ok=True)
    corpus = ref_dir / "corpus"
    print(f"  [P1] building reference: {n_graphs} live graphs "
          f"(<= {max_nodes} nodes) via the reference exporter ...")
    subprocess.run(
        [sys.executable, str(_SCRIPTS / "neo4j_eval_export.py"), "export",
         "--config", str(_KETTLE_CFG), "--out", str(corpus),
         "--num-graphs", str(n_graphs), "--max-nodes", str(max_nodes),
         "--tasks-per-graph", "1", "--seed", str(seed),
         "--sampler", "delocalized"],
        check=True, cwd=str(_ROOT), env=child_env())
    idx = corpus / "manifold_index.npz"
    print("  [P1] exporting reference manifold index (frozen baseline "
          "encoder) ...")
    subprocess.run(
        [sys.executable, "-m", "src.modelsv3.export_manifold_index",
         "--run", str(_BASELINE), "--split", "all", "--corpus", str(corpus),
         # --task -1 (ALL tasks): the exporter samples ONE random task per
         # graph, so a task-2-only index leaves task!=2 graphs with zero
         # reference rows (n_ref=0 -> spurious inf). Node embeddings are
         # task-independent; indexing all tasks only widens coverage.
         "--task", "-1", "--out", str(idx),
         "--assert-sha", "--baseline-dir", str(_BASELINE)],
        check=True, cwd=str(_ROOT), env=child_env())
    return corpus


def run_p1(ref_graphs: int, ref_max_nodes: int, ref_seed: int,
           keep: bool) -> dict:
    if not _neo4j_reachable():
        return {"gate": "P1", "pass": None, "status": "BLOCKED",
                "reason": "Neo4j unreachable; P1 requires the live DB to "
                          "build a current reference (not a fake pass)."}

    import torch

    from src.data.corpus_dataset import _build_graph_tensors  # noqa: E402
    from src.modelsv3.eval_candidate_recall import _build_encoder  # noqa: E402
    from src.modelsv3.lock_baseline import assert_encoder_sha  # noqa: E402
    from src.service.neo4j_source import Neo4jSource  # noqa: E402
    from src.service.tensor_contract import encode_subgraph  # noqa: E402

    tmp = Path(tempfile.mkdtemp(prefix="kgr_p1_"))
    try:
        corpus = _build_reference(tmp, ref_graphs, ref_max_nodes, ref_seed)
        idx = np.load(corpus / "manifold_index.npz")
        ref_emb = idx["embedding"].astype(np.float64)
        ref_gid = idx["graph_idx"]
        ref_nid = idx["neo4j_node_id"]

        cfg = json.loads((_BASELINE / "summary.json").read_text())["config"]
        from types import SimpleNamespace
        ns = SimpleNamespace(node_feat_dim=32, edge_feat_dim_schema=13,
                             node_feat_dim_schema=4, num_edge_types_max=30,
                             query_dim=18)
        assert_encoder_sha(_BASELINE, _BASELINE / "encoder.pt")
        enc = _build_encoder(cfg, ns)
        enc.load_state_dict(
            torch.load(_BASELINE / "encoder.pt", map_location="cpu"))
        enc.eval()

        sm = SchemaMap.from_yaml()
        files = sorted(corpus.glob("graph_*.npz"))
        worst = 0.0
        per_graph = []
        with Neo4jSource(schema_map=sm) as src, torch.no_grad():
            for gi, f in enumerate(files):
                with np.load(f) as gz:
                    if "neo4j_node_id" not in gz.files:
                        continue
                    nid = gz["neo4j_node_id"].astype(np.int64).tolist()
                    seed_id = (int(gz["neo4j_seed_node_id"])
                               if "neo4j_seed_node_id" in gz.files
                               else int(nid[0]))
                pull = src.pull_by_ids(nid, seed_id=seed_id)
                npz_like = encode_subgraph(pull, sm)
                g = _build_graph_tensors(npz_like)
                live = enc(g["x"], g["edge_index"], g["edge_type"],
                           g["edge_descriptor"],
                           node_descriptor=g["node_descriptor"]
                           ).node_embeddings.detach().cpu().numpy().astype(
                    np.float64)
                live_by_id = {int(i): live[r] for r, i in
                              enumerate(npz_like["neo4j_node_id"])}
                m = ref_gid == gi
                rids = ref_nid[m]
                remb = ref_emb[m]
                diffs = []
                for r in range(len(rids)):
                    lv = live_by_id.get(int(rids[r]))
                    if lv is None:
                        diffs.append(float("inf"))
                        continue
                    diffs.append(float(np.max(np.abs(lv - remb[r]))))
                gmax = max(diffs) if diffs else float("inf")
                worst = max(worst, gmax)
                per_graph.append({"graph": gi, "n_ref": int(len(rids)),
                                  "n_live": int(len(live_by_id)),
                                  "max_abs_diff": gmax})
        return {"gate": "P1", "status": "RAN",
                "pass": bool(worst < _P1_TOL and per_graph),
                "tolerance": _P1_TOL, "worst_max_abs_diff": worst,
                "n_graphs": len(per_graph), "per_graph": per_graph}
    finally:
        if keep:
            dest = _ROOT / "runs" / "service_p1_ref"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(tmp), str(dest))
            print(f"  [P1] reference kept at {dest}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main() -> int:
    ensure_pythonhashseed()  # pin determinism (re-execs once if needed)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p0-only", action="store_true")
    ap.add_argument("--ref-graphs", type=int, default=3)
    ap.add_argument("--ref-max-nodes", type=int, default=200)
    ap.add_argument("--ref-seed", type=int, default=0)
    ap.add_argument("--keep-ref", action="store_true",
                    help="keep the regenerated P1 reference for inspection")
    ap.add_argument("--json-out", type=str, default=None)
    a = ap.parse_args()

    report = {}
    print("=" * 72)
    print("KGR serving-layer verification")
    print("=" * 72)

    p0 = run_p0()
    report["P0"] = p0
    print(f"P0 schema-map no-op golden: "
          f"{'PASS' if p0['pass'] else 'FAIL'}  "
          f"({p0['n_labels']} labels, {p0['n_rel_types']} rel-types, "
          f"{len(p0['layer_mismatch'])} layer / "
          f"{len(p0['category_mismatch'])} cat mismatch)")
    if not p0["pass"]:
        print("  -> P0 FAIL: schema_map.yaml drifted from the reference. "
              "Do NOT proceed (decision tree). Mismatches:")
        print(json.dumps({"layer": p0["layer_mismatch"],
                          "category": p0["category_mismatch"]}, indent=2))

    if not a.p0_only and p0["pass"]:
        p1 = run_p1(a.ref_graphs, a.ref_max_nodes, a.ref_seed, a.keep_ref)
        report["P1"] = p1
        if p1.get("status") == "BLOCKED":
            print(f"P1 bit-exact live parity: BLOCKED ({p1['reason']})")
        else:
            print(f"P1 bit-exact live parity: "
                  f"{'PASS' if p1['pass'] else 'FAIL'}  "
                  f"(worst max|diff|={p1['worst_max_abs_diff']:.3e} "
                  f"vs tol {p1['tolerance']:.0e}, "
                  f"{p1['n_graphs']} graphs)")
            if not p1["pass"]:
                print("  -> P1 FAIL: follow the decision tree (localize the "
                      "diff to a tensor region). Per-graph:")
                print(json.dumps(p1["per_graph"], indent=2))

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(report, indent=2))
        print(f"\njson: {a.json_out}")

    ok = report["P0"]["pass"] and (
        a.p0_only or report.get("P1", {}).get("pass") in (True, None))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
