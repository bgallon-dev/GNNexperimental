r"""Three-seed, multi-arm comparison harness for KGR v3.

Runs ``train_v3.train(...)`` across seeds and model kinds and produces a
compact summary table comparing mean nDCG@10 with three-seed std.

Arms:

    v3_hyp   — KettleGraphReasonerV3 (Poincaré ball + contrastive + distance scoring)
    v3_euc   — EuclideanReasonerV3   (Euclidean + contrastive + distance scoring)

The v3-vs-v1/v2 comparison is handled by reading those arms' own
``summary.json`` files with ``--extra-baseline`` — this harness does not
re-train v1/v2; they're expected to have been run separately and their
summary paths passed in. That keeps v1/v2 reproducibility firewalled
from the v3 code.

Success criteria (per the plan):
  - v3_hyp vs v3_euc isolates the geometric claim within the new
    contrastive regime.
  - v3_hyp vs v2 (passed via --extra-baseline) isolates embedding-first
    vs score-first with the same geometry.

Usage::

    py src/modelsv3/three_seed_comparison_v3.py \
        --task 0 --seeds 0 1 2 --epochs-a 10 --epochs-b 5 \
        --out-dir runs/compare_v3_task0

Add ``--extra-baseline v2=runs/compare_task0/hyp_seed_0/summary.json \
                      runs/compare_task0/hyp_seed_1/summary.json \
                      runs/compare_task0/hyp_seed_2/summary.json`` to
fold an existing v2 run into the comparison table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.train_v3 import Config, train  # noqa: E402


_V3_KINDS = ("v3_hyp", "v3_euc")
_KIND_MODEL = {"v3_hyp": "hyperbolic", "v3_euc": "euclidean"}
_KIND_LABEL = {
    "v3_hyp": "v3 Hyperbolic (contrastive)",
    "v3_euc": "v3 Euclidean (contrastive)",
}


def _run_one(
    *,
    kind: str,
    seed: int,
    task: int,
    corpus: str,
    out: Path,
    epochs_a: int,
    epochs_b: int,
    lr: float,
    hidden_dim: int,
    num_layers: int,
    temperature: float,
    anchors: int,
    margin: float,
    cuda: bool,
    train_frac: float,
    use_tangent_approx: bool,
) -> dict:
    cfg = Config(
        corpus=corpus,
        task=task,
        model=_KIND_MODEL[kind],
        out=str(out),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        contrastive_epochs=epochs_a,
        query_epochs=epochs_b,
        lr=lr,
        seed=seed,
        cuda=cuda,
        temperature=temperature,
        anchors_per_step=anchors,
        margin=margin,
        train_frac=train_frac,
        use_tangent_approx=use_tangent_approx,
    )
    train(cfg)
    summary_path = out / "summary.json"
    with open(summary_path, "r") as f:
        return json.load(f)


def _final_ndcg10(summary: dict, task: int) -> float:
    by_t = summary.get("final_val", {}).get("by_task_type", {})
    block = by_t.get(str(task)) or by_t.get(task) or {}
    val = block.get("ndcg@10")
    if val is None:
        raise RuntimeError(
            f"summary.json has no val nDCG@10 for task {task}. "
            f"Available by_task_type keys: {list(by_t)}"
        )
    return float(val)


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "min": values[0], "max": values[0]}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def _load_extra_baselines(entries: list[str], task: int) -> dict[str, list[float]]:
    r"""``entries`` is a list of ``LABEL=PATH`` strings; multiple paths
    for the same LABEL are pooled. Each PATH is a summary.json from a
    prior v1/v2 run."""
    out: dict[str, list[float]] = {}
    for e in entries:
        if "=" not in e:
            raise ValueError(
                f"--extra-baseline entry must be LABEL=PATH; got {e!r}"
            )
        label, path = e.split("=", 1)
        with open(path) as f:
            summary = json.load(f)
        out.setdefault(label, []).append(_final_ndcg10(summary, task))
    return out


def _print_table(results: dict) -> None:
    print("\n" + "=" * 88)
    print("KGR v3 MULTI-ARM COMPARISON")
    print("=" * 88)
    print(
        f"Task: {results['task']}   seeds: {results['seeds']}   "
        f"epochs (A/B): {results['epochs_a']}/{results['epochs_b']}"
    )
    print(f"hidden={results['hidden_dim']}  layers={results['num_layers']}  lr={results['lr']}")
    print()
    print(f"{'Arm':<34} {'Mean nDCG@10':>14} {'Std':>8} {'Min':>8} {'Max':>8}  {'Seeds':>6}")
    print("-" * 82)
    for kind in results["v3_kinds"]:
        s = results[f"{kind}_summary"]
        label = _KIND_LABEL.get(kind, kind)
        n = len(results[f"{kind}_ndcgs"])
        print(
            f"{label:<34} {s['mean']:>14.4f} {s['std']:>8.4f} "
            f"{s['min']:>8.4f} {s['max']:>8.4f}  {n:>6d}"
        )
    for label, stats in results.get("extra_baselines_summary", {}).items():
        n = len(results["extra_baselines_ndcgs"][label])
        print(
            f"{label:<34} {stats['mean']:>14.4f} {stats['std']:>8.4f} "
            f"{stats['min']:>8.4f} {stats['max']:>8.4f}  {n:>6d}"
        )

    # Pairwise deltas against v3_hyp (if present).
    print()
    if "v3_hyp" in results["v3_kinds"]:
        hyp = results["v3_hyp_summary"]
        for kind in results["v3_kinds"]:
            if kind == "v3_hyp":
                continue
            o = results[f"{kind}_summary"]
            diff = hyp["mean"] - o["mean"]
            pooled = max(hyp["std"], o["std"])
            _report_pair(f"v3_hyp − {_KIND_LABEL.get(kind, kind)}", diff, pooled)
        for label, s in results.get("extra_baselines_summary", {}).items():
            diff = hyp["mean"] - s["mean"]
            pooled = max(hyp["std"], s["std"])
            _report_pair(f"v3_hyp − {label}", diff, pooled)


def _report_pair(header: str, diff: float, pooled_std: float) -> None:
    print(f"{header}: {diff:+.4f}")
    if pooled_std == 0.0:
        print("  (single seed; no noise estimate)")
    elif abs(diff) < pooled_std:
        print("  gap smaller than 1 std -- within seed noise")
    elif abs(diff) < 2 * pooled_std:
        print("  gap 1-2 std -- suggestive, not conclusive")
    else:
        direction = "LHS" if diff > 0 else "RHS"
        print(f"  gap > 2 std -- {direction} wins with confidence")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--epochs-a", type=int, default=10)
    p.add_argument("--epochs-b", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--anchors", type=int, default=64)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--train-frac", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--v3-kinds", type=str, nargs="+", default=list(_V3_KINDS), choices=_V3_KINDS,
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--use-tangent-approx", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="Reuse an existing summary.json instead of re-training.")
    p.add_argument(
        "--extra-baseline", type=str, nargs="*", default=[],
        help="Pre-existing v1/v2 summary paths to fold in. Format: LABEL=PATH. "
             "Multiple PATHs per LABEL (one per seed) are averaged.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ndcgs: dict[str, list[float]] = {k: [] for k in args.v3_kinds}
    for seed in args.seeds:
        for kind in args.v3_kinds:
            run_out = out_dir / f"{kind}_seed_{seed}"
            summary_path = run_out / "summary.json"
            if args.skip_existing and summary_path.exists():
                print(f"[skip-existing] reusing {summary_path}")
                with open(summary_path) as f:
                    summary = json.load(f)
            else:
                print(f"\n{'#' * 88}")
                print(f"# {_KIND_LABEL[kind]}  seed={seed}")
                print(f"{'#' * 88}")
                summary = _run_one(
                    kind=kind, seed=seed, task=args.task, corpus=args.corpus,
                    out=run_out, epochs_a=args.epochs_a, epochs_b=args.epochs_b,
                    lr=args.lr, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                    temperature=args.temperature, anchors=args.anchors,
                    margin=args.margin, cuda=args.cuda, train_frac=args.train_frac,
                    use_tangent_approx=args.use_tangent_approx,
                )
            ndcgs[kind].append(_final_ndcg10(summary, args.task))

    extra = _load_extra_baselines(args.extra_baseline, args.task)

    results = {
        "task": args.task,
        "seeds": args.seeds,
        "epochs_a": args.epochs_a,
        "epochs_b": args.epochs_b,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "lr": args.lr,
        "v3_kinds": args.v3_kinds,
    }
    for kind in args.v3_kinds:
        results[f"{kind}_ndcgs"] = ndcgs[kind]
        results[f"{kind}_summary"] = _summarize(ndcgs[kind])
    if extra:
        results["extra_baselines_ndcgs"] = extra
        results["extra_baselines_summary"] = {k: _summarize(v) for k, v in extra.items()}

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    _print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
