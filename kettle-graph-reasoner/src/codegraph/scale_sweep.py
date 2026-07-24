r"""Width-scaling sweep against the tiny-by-design ceiling.

For each cell ``(hidden_dim, num_layers=4)``:

  1. Stage-A pretrain the v3 hyperbolic encoder on the 6-repo code
     corpus (``code_v1``) via ``train_v3`` with the locked stability
     recipe (small-gain Xavier + tangent_scale + radial-reg decay).
  2. LORO-CV eval over the same repos via ``src.codegraph.harness``,
     using the freshly trained encoder.
  3. Pluck the headline numbers into ``scaling_curve.json``.

Cells, sweep axis, success criterion (pool nDCG@10 monotone + crosses
anchor-BFS at the largest cell), seed count are all locked upstream.
Resumable: cells whose ``encoder.pt`` already exist are not retrained.

Run from ``kettle-graph-reasoner/`` with the DML venv's python so the
``privateuseone:0`` device is registered:

    .venv_dml\Scripts\python.exe -m src.codegraph.scale_sweep \
        --corpus-root ../corpus_validation \
        --out runs/scale_sweep --device privateuseone:0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Optional: register torch-directml so torch.device("privateuseone:0") resolves.
try:  # pragma: no cover
    import torch_directml  # noqa: F401
except ImportError:
    pass

import torch

from ..modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3


STABILITY_FLAGS = [
    "--tangent-scale", "0.1",
    "--radial-reg-weight", "0.01",
    "--radial-reg-weight-end", "0.001",
    "--temperature", "1.0",
    "--margin", "0.5",
    "--curvature", "1.0",
    "--type-dim", "8",
    "--lr", "0.0003",
    "--positive-mix", "0.5",
    "--neighbor-exclude-k", "1",
    "--anchors-per-step", "64",
    "--train-graphs-frac", "1.0",
    "--stage-b-loss", "pairwise",
    # The intrinsic probe runs silhouette_score (O(N^2) pairwise distances)
    # on a val graph after the training loop and before encoder.pt save.
    # On code corpora where val graphs can be 300k+ nodes, this hangs the
    # subprocess for hours. Disable it; we don't use the metric anyway.
    "--no-intrinsic",
]


def _count_params(hidden_dim: int, num_layers: int) -> int:
    enc = KettleGraphReasonerV3(
        node_feat_dim=32, edge_feat_dim=13,
        hidden_dim=hidden_dim, num_layers=num_layers, type_dim=8,
        num_edge_types_max=30, node_feat_dim_schema=4,
        tangent_scale_init=0.1,
    )
    return sum(p.numel() for p in enc.parameters())


def _stage_a(cell_dir: Path, hidden: int, layers: int, corpus: str,
             epochs: int, seed: int, device: str, model_kind: str,
             n_neg_sample: int = 0,
             no_autocast: bool = False,
             early_stop: bool = False) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "src.training.train_v3",
        "--corpus", corpus,
        "--task", "0",
        "--model", model_kind,
        "--out", str(cell_dir),
        "--hidden-dim", str(hidden),
        "--num-layers", str(layers),
        "--seed", str(seed),
        "--contrastive-epochs", str(epochs),
        "--query-epochs", "0",
        "--device", device,
        *STABILITY_FLAGS,
    ]
    if n_neg_sample > 0:
        cmd += ["--stage-a-n-neg-sample", str(n_neg_sample)]
    if no_autocast:
        cmd += ["--no-autocast"]
    if early_stop:
        cmd += ["--early-stop"]
    # If a previous run got killed mid-Stage-A, an atomic stage_a_state.pt
    # is on disk but encoder.pt is not — auto-resume from there.
    state = cell_dir / "stage_a_state.pt"
    if state.exists():
        cmd += ["--resume-from", str(state)]
        print(f"  [resume] {state} found; appending --resume-from")
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _eval(ckpt_dir: Path, eval_dir: Path, corpus_root: str,
          head_epochs: int, device: str, folds: int,
          repo_split: str = "", max_cases_per_task: int = 0,
          task_families: list[str] | None = None,
          max_eval_cases_per_task_per_repo: int = 0,
          corpus_format: str = "jsonl") -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "src.codegraph.harness",
        "--corpus-root", corpus_root,
        "--ckpt", str(ckpt_dir),
        "--out", str(eval_dir),
        "--epochs", str(head_epochs),
        "--device", device,
    ]
    if repo_split:
        cmd += ["--repo-split", repo_split]
    elif folds > 0:
        cmd += ["--folds", str(folds)]
    if max_cases_per_task > 0:
        cmd += ["--max-cases-per-task", str(max_cases_per_task)]
    if max_eval_cases_per_task_per_repo > 0:
        cmd += ["--max-eval-cases-per-task-per-repo",
                str(max_eval_cases_per_task_per_repo)]
    if task_families:
        cmd += ["--task-families", *task_families]
    if corpus_format and corpus_format != "jsonl":
        cmd += ["--corpus-format", corpus_format]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return json.loads((eval_dir / "report.json").read_text())


def _row_from_report(report: dict) -> dict:
    m = report["model"].get("cv_by_split_mode", {})
    a = report["baseline_anchor"].get("cv_by_split_mode", {})
    r = report["baseline_random"].get("cv_by_split_mode", {})
    # De-localized pool numbers (nonlocal = answer not adjacent to anchor):
    # the fair "beats heuristic" cells, isolated from the adjacent-answer
    # cases where anchor-BFS is near-oracle by construction.
    ml = report["model"].get("cv_by_locality_mode", {})
    al = report["baseline_anchor"].get("cv_by_locality_mode", {})
    return {
        "model_test_within_ndcg10": m.get("test|within", {}).get("ndcg@10"),
        "model_test_pool_ndcg10":   m.get("test|pool", {}).get("ndcg@10"),
        "model_test_pool_mrr":      m.get("test|pool", {}).get("mrr"),
        "model_test_pool_r10":      m.get("test|pool", {}).get("r@10"),
        "anchor_test_within_ndcg10": a.get("test|within", {}).get("ndcg@10"),
        "anchor_test_pool_ndcg10":   a.get("test|pool", {}).get("ndcg@10"),
        "random_test_pool_ndcg10":   r.get("test|pool", {}).get("ndcg@10"),
        # de-localized (nonlocal) pool headline
        "model_test_pool_nonlocal_ndcg10":
            ml.get("test|pool|nonlocal", {}).get("ndcg@10"),
        "anchor_test_pool_nonlocal_ndcg10":
            al.get("test|pool|nonlocal", {}).get("ndcg@10"),
        "model_test_within_nonlocal_ndcg10":
            ml.get("test|within|nonlocal", {}).get("ndcg@10"),
        "anchor_test_within_nonlocal_ndcg10":
            al.get("test|within|nonlocal", {}).get("ndcg@10"),
        "folds": m.get("test|pool", {}).get("folds"),
        "node_emb_norm_mean": _mean_h(report),
    }


def _mean_h(report: dict) -> float | None:
    return None  # populated post-hoc if we wire it through; not required


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="128,256,512,1024",
                    help="comma-separated hidden_dim values; layers=4 fixed")
    ap.add_argument("--corpus", default="src/data/corpus/code_v1",
                    help="Stage-A corpus dir (per-repo NPZs)")
    ap.add_argument("--corpus-root", default="../corpus_validation",
                    help="LORO-CV eval source (raw repo dirs)")
    ap.add_argument("--out", default="runs/scale_sweep")
    ap.add_argument("--epochs", type=int, default=10,
                    help="Stage-A contrastive epochs per cell")
    ap.add_argument("--head-epochs", type=int, default=10,
                    help="LORO-CV head training epochs")
    ap.add_argument("--folds", type=int, default=0,
                    help="cap LORO folds (0 = all repos)")
    ap.add_argument("--seed", type=int, default=0,
                    help="(single-seed mode) seed for Stage-A + head init")
    ap.add_argument("--seeds", default="",
                    help="comma-separated seed list; overrides --seed. "
                    "Produces mean±std per cell in scaling_curve.json. "
                    "Cells loop on the inside, seeds on the outside so a "
                    "disconnect mid-sweep loses at most one cell×seed.")
    ap.add_argument("--device", default="privateuseone:0")
    ap.add_argument("--model", default="hyperbolic",
                    choices=["hyperbolic", "euclidean"])
    ap.add_argument("--force", action="store_true",
                    help="retrain cells whose encoder.pt exists")
    ap.add_argument("--smoke", action="store_true",
                    help="single small cell, 1 epoch each, sanity check")
    ap.add_argument("--repo-split", default="",
                    help="forwarded to harness; path to train/test JSON. "
                    "Use category-stratified split for 60-repo runs.")
    ap.add_argument("--stage-a-n-neg-sample", type=int, default=0,
                    help="forwarded to train_v3; sampled InfoNCE K. "
                    "Required (>0) for fitting large graphs at h>=512.")
    ap.add_argument("--max-cases-per-task", type=int, default=0,
                    help="forwarded to harness; cap training cases per "
                    "task per fold. Required at v0.2-scale corpus to "
                    "avoid week-long sweeps.")
    ap.add_argument("--max-eval-cases-per-task-per-repo", type=int, default=0,
                    help="forwarded to harness; cap eval cases per "
                    "(held-out repo, task). Required at v0.2-scale "
                    "corpus — without it, eval over scipy/django/pandas "
                    "is the wall-time killer.")
    ap.add_argument("--corpus-format", default="jsonl",
                    choices=["jsonl", "pack"],
                    help="forwarded to harness; 'pack' loads from the "
                    "kgr_pack v0.2 binary corpus instead of per-repo "
                    "jsonl. ~45x smaller on disk, ~10x faster ingest.")
    ap.add_argument("--task-families", nargs="+",
                    default=["ranking", "classification"],
                    help="forwarded to harness; default drops "
                    "abstain_ranking (known-degenerate in v0.2 — "
                    "head trivially learns to always abstain).")
    ap.add_argument("--no-autocast", action="store_true",
                    help="forwarded to train_v3; disable bf16 autocast "
                    "around the encoder forward (Patch 2). Default off "
                    "on CUDA = autocast enabled.")
    ap.add_argument("--early-stop", action="store_true",
                    help="forwarded to train_v3; early-stop Stage-A "
                    "when the InfoNCE gap plateaus (Patch 3).")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        cells = [128]
        args.epochs = 1
        args.head_epochs = 1
    else:
        cells = [int(c) for c in args.cells.split(",")]

    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed]

    rows: list[dict] = []
    t0 = time.time()
    # Cells inside, seeds outside: a disconnect mid-run loses at most
    # one (cell, seed); the rest is resumable via the encoder.pt /
    # report.json existence checks.
    for seed in seeds:
        for h in cells:
            cell = out / f"h{h}_l4_seed{seed}"
            ev = out / f"h{h}_l4_seed{seed}_eval"
            params = _count_params(h, 4)
            print(f"\n=== cell h={h} L=4 seed={seed}  params={params:,}  "
                  f"({(time.time()-t0)/60:.1f} min elapsed) ===")

            if (cell / "encoder.pt").exists() and not args.force:
                print(f"  [skip stage-A] {cell}/encoder.pt exists")
            else:
                _stage_a(cell, h, 4, args.corpus, args.epochs, seed,
                         args.device, args.model, args.stage_a_n_neg_sample,
                         no_autocast=args.no_autocast,
                         early_stop=args.early_stop)

            if (ev / "report.json").exists() and not args.force:
                print(f"  [skip eval]    {ev}/report.json exists")
                report = json.loads((ev / "report.json").read_text())
            else:
                report = _eval(cell, ev, args.corpus_root,
                               args.head_epochs, args.device, args.folds,
                               args.repo_split,
                               max_cases_per_task=args.max_cases_per_task,
                               task_families=args.task_families,
                               max_eval_cases_per_task_per_repo=args.max_eval_cases_per_task_per_repo,
                               corpus_format=args.corpus_format)

            row = {"hidden": h, "layers": 4, "seed": seed, "params": params,
                   **_row_from_report(report)}
            rows.append(row)
            print("  -> "
                  f"within={row['model_test_within_ndcg10']:.3f} "
                  f"pool={row['model_test_pool_ndcg10']:.3f} "
                  f"(anchor pool={row['anchor_test_pool_ndcg10']:.3f})")

    # Aggregate across seeds per cell.
    agg = _aggregate(rows)
    out_json = out / "scaling_curve.json"
    out_json.write_text(json.dumps({
        "config": vars(args), "rows": rows, "aggregated": agg,
        "wall_minutes": (time.time() - t0) / 60.0,
    }, indent=2))
    print("\n=== scaling curve (per-seed) ===")
    cols = ["hidden", "seed", "params", "model_test_within_ndcg10",
            "model_test_pool_ndcg10", "anchor_test_pool_ndcg10",
            "random_test_pool_ndcg10"]
    _print_row(cols, header=True)
    for r in rows:
        _print_row(cols, r=r)

    if len(seeds) > 1:
        print("\n=== aggregated (mean ± std across seeds) ===")
        _print_agg(agg)
    print(f"\nreport: {out_json}")


def _aggregate(rows: list[dict]) -> list[dict]:
    """Group rows by hidden_dim; compute mean / std / min / max across
    seeds for the headline metrics."""
    from collections import defaultdict
    keys = ("model_test_within_ndcg10", "model_test_pool_ndcg10",
            "model_test_pool_mrr", "model_test_pool_r10",
            "anchor_test_pool_ndcg10",
            "model_test_pool_nonlocal_ndcg10",
            "anchor_test_pool_nonlocal_ndcg10",
            "model_test_within_nonlocal_ndcg10",
            "anchor_test_within_nonlocal_ndcg10")
    by_h: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_h[r["hidden"]].append(r)
    out: list[dict] = []
    for h in sorted(by_h):
        cell_rows = by_h[h]
        agg: dict = {"hidden": h, "n_seeds": len(cell_rows),
                     "params": cell_rows[0]["params"]}
        for k in keys:
            vals = [r[k] for r in cell_rows if r.get(k) is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
            std = var ** 0.5
            agg[f"{k}_mean"] = mean
            agg[f"{k}_std"] = std
            agg[f"{k}_min"] = min(vals)
            agg[f"{k}_max"] = max(vals)
        out.append(agg)
    return out


def _print_row(cols: list[str], r: dict | None = None,
               header: bool = False) -> None:
    if header:
        print(" ".join(f"{c:>30}" for c in cols))
        return
    assert r is not None
    pieces = []
    for c in cols:
        v = r.get(c)
        if c == "params":
            pieces.append(f"{v:>30,}")
        elif c in ("hidden", "seed"):
            pieces.append(f"{v:>30}")
        elif v is None:
            pieces.append(f"{'—':>30}")
        else:
            pieces.append(f"{v:>30.3f}")
    print(" ".join(pieces))


def _print_agg(agg: list[dict]) -> None:
    print(f"{'hidden':>8} {'params':>10} {'n':>3}  "
          f"{'within (mean±std)':>22}  {'pool (mean±std)':>22}  "
          f"{'anchor pool':>12}")
    for a in agg:
        wm = a.get("model_test_within_ndcg10_mean")
        ws = a.get("model_test_within_ndcg10_std", 0.0)
        pm = a.get("model_test_pool_ndcg10_mean")
        ps = a.get("model_test_pool_ndcg10_std", 0.0)
        am = a.get("anchor_test_pool_ndcg10_mean", 0.0)
        print(f"{a['hidden']:>8} {a['params']:>10,} {a['n_seeds']:>3}  "
              f"{wm:>10.3f} ± {ws:<8.3f}  "
              f"{pm:>10.3f} ± {ps:<8.3f}  "
              f"{am:>12.3f}")


if __name__ == "__main__":
    main()
