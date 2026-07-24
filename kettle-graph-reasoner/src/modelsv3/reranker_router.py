r"""Per-task-family reranker router (v3.1.x deployment consolidation).

ZERO new training. v3.3 established that the optimal reranker recipe is
task-family-dependent: the v3.2 *damped* residual (z(retr)+scale*(v2-0.5))
is best for the locality-friendly temporal task (deployed 0.672), while
the v3.3 *blend* (a*z(retr|cand)+b*z(logit v2|cand)) is best for the
geometry-sensitive tasks (beats the WS3 per-task v2 on 0/4/5). This
module ships the best measured system by *selecting the recipe per task*
-- the existing per-cell validation gate (residual-vs-retriever) lifted
one level to also choose recipe-vs-recipe.

It is a pure post-hoc aggregator over the two finished sweeps
(``runs/sweep_reranker_v32`` and ``runs/sweep_reranker_v33``); it trains
nothing and re-measures nothing. Selection rule, per task:

  routed recipe = argmax over {v3.2-damped, v3.3-blend} of the
                  3-seed-mean val-gated deployed ndcg@10.

Each underlying deployed number is already validation-gated to be
>= its retriever (the reranker_v32 do-no-harm gate) and the retriever
baseline is *bit-identical* across the two sweeps (asserted), so the
per-task max is still >= retriever -> the routed system provably never
regresses vs the retriever. The data-driven choice is additionally
cross-checked against the geometry-family heuristic (geom -> blend,
else -> damped); agreement means the choice is not noise-fitting.

Honest scope: selection is on the same val set every deployed number in
this study uses (there is no separate held-out test split in the hybrid
harness); the 3-seed spread is the noise indicator. Reported as such.

Usage
-----
    py -m src.modelsv3.reranker_router \
        --v32-results runs/sweep_reranker_v32/sweep_reranker_v32_results.json \
        --v33-results runs/sweep_reranker_v33/sweep_reranker_v32_results.json \
        --config src/modelsv3/sweep_config_reranker_v33.json \
        --out runs/reranker_router
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

V32_NAME = "v3.2-damped"
V33_NAME = "v3.3-blend"


def _oracle_by_task(cells: dict) -> dict:
    """Per-task mean oracle ndcg@10 (top-C ceiling) from cells, for
    contextualising each gap honestly (e.g. task1's ~0.26 ceiling)."""
    acc: dict[int, list] = {}
    for v in cells.values():
        if v.get("state") == "complete" and "oracle_ndcg@10" in v:
            acc.setdefault(int(v["task"]), []).append(
                float(v["oracle_ndcg@10"]))
    return {t: (sum(o) / len(o)) for t, o in acc.items() if o}


def _seed_deploys(cells: dict, task: int) -> list[float]:
    return [float(v["hybrid_ndcg@10"]) for v in cells.values()
            if v.get("state") == "complete" and int(v["task"]) == task]


def route(v32: dict, v33: dict, extra: dict[str, dict] | None = None) -> dict:
    """``extra`` maps arm-name -> results dict with
    ``{"by_task": {"<task>": {"deployed_ndcg@10_mean": float}}}`` —
    additional zero-training gate-set arms (e.g. **anchor-BFS**, per the
    2026-07 deployed-routing eval where BFS beat the production confidence
    sort on every family). An extra arm is chosen for a task ONLY if it
    strictly beats both recipes there AND >= the retriever (the same
    do-no-harm gate); reporting fields that require sweep cells fall back
    to the best v3.x recipe's values."""
    b32 = v32["gate"]["by_task"]
    b33 = v33["gate"]["by_task"]
    tasks = sorted(b32.keys(), key=int)
    if sorted(b33.keys(), key=int) != tasks:
        raise SystemExit("[router] ABORT: task sets differ between the "
                         "two sweeps -- not comparable.")
    orc = _oracle_by_task(v33.get("cells", {}))

    by_task: dict[str, dict] = {}
    any_regr = False
    for t in tasks:
        r32, r33 = b32[t], b33[t]
        a = float(r32["v31_ndcg@10_mean"])
        b = float(r33["v31_ndcg@10_mean"])
        if abs(a - b) > 1e-6:
            raise SystemExit(
                f"[router] ABORT: task{t} retriever baseline differs "
                f"(v32={a:.6f} v33={b:.6f}); the two sweeps are not "
                f"apples-to-apples, refusing to route.")
        retr = a
        cand = {V32_NAME: r32, V33_NAME: r33}
        # argmax 3-seed-mean deployed ndcg@10; tie -> simpler/older v3.2.
        chosen = max((V33_NAME, V32_NAME),
                     key=lambda nm: cand[nm]["hybrid_ndcg@10_mean"])
        if (cand[V32_NAME]["hybrid_ndcg@10_mean"]
                >= cand[V33_NAME]["hybrid_ndcg@10_mean"]):
            chosen = V32_NAME
        c = cand[chosen]
        dep = float(c["hybrid_ndcg@10_mean"])
        extra_means: dict[str, float] = {}
        base_chosen = chosen
        for nm, arm in (extra or {}).items():
            bt = arm.get("by_task", {})
            if t in bt:
                m = float(bt[t]["deployed_ndcg@10_mean"])
                extra_means[nm] = m
                if m > dep and m >= retr - 1e-9:
                    chosen, dep = nm, m
        if dep < retr - 1e-9:
            raise SystemExit(
                f"[router] ABORT: task{t} routed deployed {dep:.6f} < "
                f"retriever {retr:.6f}; the no-regression invariant is "
                f"violated -- investigate the per-cell gate.")
        regr = bool(c["regression"])
        any_regr = any_regr or regr
        is_geom = bool(c["is_geometry_sensitive"])
        heuristic = V33_NAME if is_geom else V32_NAME
        seeds = _seed_deploys(
            (v33 if base_chosen == V33_NAME else v32).get("cells", {}),
            int(t)) if chosen == base_chosen else []
        by_task[t] = {
            "is_geometry_sensitive": is_geom,
            "retriever_ndcg@10_mean": retr,
            "oracle_ndcg@10_mean": orc.get(int(t)),
            "v32_deployed_mean": float(r32["hybrid_ndcg@10_mean"]),
            "v33_deployed_mean": float(r33["hybrid_ndcg@10_mean"]),
            "chosen_recipe": chosen,
            "routed_deployed_mean": dep,
            "routed_deployed_std": (st.pstdev(seeds)
                                    if len(seeds) > 1 else 0.0),
            "routed_gap_closed_frac": c.get("gap_closed_frac_mean"),
            "ws3_pertask_gap_closed": c.get("ws3_pertask_gap_closed"),
            "beats_ws3": bool(c["beats_ws3"]),
            "regression": regr,
            "heuristic_recipe": heuristic,
            "data_matches_heuristic": chosen == heuristic,
            "extra_arm_means": extra_means,
        }

    geom = [v for v in by_task.values() if v["is_geometry_sensitive"]]
    geom_beat = [v for v in geom if v["beats_ws3"]]

    def _macro(sel) -> dict:
        allv = [sel(t) for t in by_task]
        gv = [sel(t) for t in by_task
              if by_task[t]["is_geometry_sensitive"]]
        # geom subset can be empty (e.g. a single non-geometry task such
        # as the real-temporal head) -> nan, not a crash.
        return {"all_tasks_mean": (st.mean(allv) if allv
                                   else float("nan")),
                "geom_tasks_mean": (st.mean(gv) if gv
                                    else float("nan"))}

    systems = {
        "retriever_only": _macro(
            lambda t: by_task[t]["retriever_ndcg@10_mean"]),
        "v32_only": _macro(lambda t: by_task[t]["v32_deployed_mean"]),
        "v33_only": _macro(lambda t: by_task[t]["v33_deployed_mean"]),
        "router": _macro(lambda t: by_task[t]["routed_deployed_mean"]),
    }
    return {
        "by_task": by_task,
        "systems": systems,
        "geom_tasks_beating_ws3": len(geom_beat),
        "geom_tasks_total": len(geom),
        "any_regression": any_regr,
        "acceptance_pass": (len(geom_beat) == len(geom)
                            and len(geom) > 0 and not any_regr),
        "all_data_choices_match_heuristic": all(
            v["data_matches_heuristic"] for v in by_task.values()),
    }


def _print_report(rep: dict, out: Path) -> None:
    print()
    print("=" * 100)
    print("Per-task-family reranker ROUTER  (zero training; selects "
          "v3.2-damped vs v3.3-blend per task)")
    print("=" * 100)
    print(f"  {'task':<5}{'geom':>5}{'retr':>8}{'oracle':>8}"
          f"{'v32dep':>8}{'v33dep':>8}{'->recipe':>13}{'routed':>8}"
          f"{'+-std':>7}{'gap':>7}{'WS3':>7}{'beats':>6}{'regr':>5}{'h?':>4}")
    for t, v in rep["by_task"].items():
        oc = v["oracle_ndcg@10_mean"]
        ocs = "n/a" if oc is None else f"{oc:.3f}"
        g = v["routed_gap_closed_frac"]
        gs = "n/a" if g is None else f"{g:+.2f}"
        w = v["ws3_pertask_gap_closed"]
        ws = "n/a" if w is None else f"{w:+.2f}"
        print(f"  {t:<5}{('Y' if v['is_geometry_sensitive'] else '-'):>5}"
              f"{v['retriever_ndcg@10_mean']:>8.3f}{ocs:>8}"
              f"{v['v32_deployed_mean']:>8.3f}{v['v33_deployed_mean']:>8.3f}"
              f"{v['chosen_recipe']:>13}{v['routed_deployed_mean']:>8.3f}"
              f"{v['routed_deployed_std']:>7.3f}{gs:>7}{ws:>7}"
              f"{('YES' if v['beats_ws3'] else 'no'):>6}"
              f"{('YES' if v['regression'] else 'no'):>5}"
              f"{('=' if v['data_matches_heuristic'] else 'X'):>4}")
    s = rep["systems"]
    print()
    print("  deployable system macro-means (mean of per-task deployed "
          "ndcg@10):")
    print(f"    {'system':<16}{'all 6 tasks':>14}{'4 geom tasks':>15}")
    for nm in ("retriever_only", "v32_only", "v33_only", "router"):
        print(f"    {nm:<16}{s[nm]['all_tasks_mean']:>14.4f}"
              f"{s[nm]['geom_tasks_mean']:>15.4f}")
    print()
    print(f"  geom tasks beating WS3 (routed): "
          f"{rep['geom_tasks_beating_ws3']}/{rep['geom_tasks_total']}"
          f"  | any regression: {rep['any_regression']}"
          f"  | acceptance: {rep['acceptance_pass']}"
          f"  | data==heuristic on every task: "
          f"{rep['all_data_choices_match_heuristic']}")
    print("  h? column: '=' the data-driven choice equals the "
          "geometry-family heuristic (geom->blend, else->damped); "
          "'X' they disagree.")
    print("  Honest notes: (1) router >= max(v32_only, v33_only) per "
          "task BY CONSTRUCTION (per-task argmax) and >= retriever "
          "(per-cell do-no-harm gate) -- it provably degrades nothing. "
          "(2) Selection is on the study's val set (no separate test "
          "split exists here); the +-std is the 3-seed noise floor. "
          "(3) task1 'beats_ws3' is noise-on-noise (oracle ceiling "
          "~0.26, all numbers near zero) -- not a real win. (4) "
          "Acceptance stays False iff a geom task (task3 multihop) "
          "still trails WS3; the router does NOT manufacture a pass, "
          "it ships the best measured recipe per task and resolves the "
          "temporal trade-back by deployment choice.")
    print(f"\n  results: {out / 'router_results.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v32-results", type=str,
                    default="runs/sweep_reranker_v32/"
                            "sweep_reranker_v32_results.json")
    ap.add_argument("--v33-results", type=str,
                    default="runs/sweep_reranker_v33/"
                            "sweep_reranker_v32_results.json")
    ap.add_argument("--config", type=str,
                    default="src/modelsv3/sweep_config_reranker_v33.json")
    ap.add_argument("--out", type=str, default="runs/reranker_router")
    ap.add_argument("--extra-arm", action="append", default=[],
                    metavar="NAME=RESULTS.json",
                    help="additional zero-training gate-set arm (e.g. "
                         "bfs=runs/bfs_arm.json); JSON schema: "
                         "{'by_task': {'<t>': {'deployed_ndcg@10_mean': x}}}")
    a = ap.parse_args()

    v32 = json.loads(Path(a.v32_results).read_text())
    v33 = json.loads(Path(a.v33_results).read_text())
    extra = {}
    for spec in a.extra_arm:
        nm, _, pth = spec.partition("=")
        extra[nm] = json.loads(Path(pth).read_text())
    rep = route(v32, v33, extra=extra or None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "v32_results": a.v32_results,
        "v33_results": a.v33_results,
        "recipe_map": {V32_NAME: "reranker_v32 --combine-mode v32",
                       V33_NAME: "reranker_v32 --combine-mode blend"},
        "router": rep,
    }
    (out / "router_results.json").write_text(json.dumps(payload, indent=2))
    _print_report(rep, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
