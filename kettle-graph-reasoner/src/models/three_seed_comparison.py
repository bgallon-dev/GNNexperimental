r"""Three-seed training comparison: hyp vs euc (clean) vs euc_plus (MLP head).

Extends the earlier ``three_seed_comparison.py`` to support a third model
(EuclideanPlusBaseline with an MLP scoring head), closing the scoring-head
confound from the task-2 result.

The three-way comparison tests:

  hyp       = KettleGraphReasonerClean (hyperbolic + distance scoring)
  euc       = EuclideanBaselineClean   (Euclidean + distance scoring)  [control]
  euc_plus  = EuclideanPlusBaseline    (Euclidean + MLP scoring head)

The earlier task-2 result showed hyp wins over euc by ~0.45 nDCG. That gap
could be attributed to:
  (A) hyperbolic geometry preventing a degenerate-shell solution, or
  (B) the MLP scoring head fundamentally being more expressive than
      distance-based scoring.

``euc_plus`` isolates hypothesis B. If hyp wins over euc but ties / loses
to euc_plus, the advantage was scoring-head-specific, not geometric. If
hyp wins over both, hyperbolic geometry is doing work beyond just avoiding
degenerate shells.

Implementation notes
--------------------
``train_clean.py`` reads ``out.node_logits`` and ``model.c`` in its
diagnostic print statements. ``EuclideanPlusBaseline`` returns a
``KGROutput`` (no ``node_logits``) and has no ``.c`` attribute. Rather
than modifying ``train_clean.py``, this script patches its diagnostic
helpers at runtime when ``model_kind == 'euc_plus'``. Core training and
metrics are unchanged; only the stdout telemetry is made robust to the
missing fields.

Run::

    py three_seed_comparison_v2.py --task 2 --seeds 0 1 2 --epochs 20 \
        --models hyp euc euc_plus --out-dir runs/compare_v2_task2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")


# ---- Model kinds and their class resolution ------------------------------
_MODEL_KINDS = ("hyp", "euc", "euc_plus")

_MODEL_LABELS = {
    "hyp": "Hyperbolic (clean)",
    "euc": "Euclidean (clean)",
    "euc_plus": "Euclidean+ (MLP head)",
}


def _build_cfg(
    *,
    task: int,
    corpus: str,
    epochs: int,
    hidden_dim: int,
    num_layers: int,
    lr: float,
    out: str,
    seed: int,
    cuda: bool,
    model_kind: str,
) -> argparse.Namespace:
    """Build an argparse-compatible Namespace for train_clean.train()."""
    return argparse.Namespace(
        task=task,
        corpus=corpus,
        epochs=epochs,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        lr=lr,
        lr_schedule="cosine",
        lr_step_epoch=15,
        early_stop_patience=0,
        edge_loss_weight=0.5,
        tangent_scale=0.10,
        seed=seed,
        log_every=500,
        out=out,
        cuda=cuda,
        model_kind=model_kind,
    )


def _patched_distance_stats(out) -> dict:
    r"""Distance-stats replacement for models whose output has no
    ``node_logits``. ``EuclideanPlusBaseline`` produces ``node_scores``
    via ``sigmoid(MLP([h, q]))``, so the raw scores contain the ranking
    signal directly. We read spread off ``node_scores`` as a diagnostic
    of how rankable the model's outputs are.

    The resulting ``dist_*`` values are NOT interpretable as distances
    for euc_plus — they're the score distribution. The key comparable
    quantity is ``dist_std``: low value means near-constant outputs
    (degenerate / non-rankable), high value means the model is producing
    a spread of scores. That's the mechanism signal we care about across
    both scoring heads.
    """
    # Use node_scores (sigmoid outputs, in [0,1]) as the ranking-signal proxy.
    s = out.node_scores.detach()
    return {
        "dist_mean": float(s.mean()),
        "dist_std": float(s.std(unbiased=False)) if s.numel() > 1 else 0.0,
        "dist_min": float(s.min()),
        "dist_max": float(s.max()),
    }


def _patched_embedding_norm_stats(h, c):
    r"""Embedding-norm stats for models with no curvature. ``c`` is passed
    but ignored; ``boundary`` is reported as NaN (not meaningful in flat
    space)."""
    h = h.detach()
    norms = h.norm(dim=-1)
    return {
        "mean_norm": float(norms.mean()),
        "max_norm": float(norms.max()),
        "min_norm": float(norms.min()),
        "std_norm": float(norms.std(unbiased=False)) if norms.numel() > 1 else 0.0,
        "boundary": float("nan"),  # no ball, no boundary
    }


class _DummyC:
    r"""Stand-in for ``model.c`` on Euclidean models. ``train_clean.py``
    calls ``model.c.clamp_min(...).sqrt()`` when computing the
    ``boundary`` diagnostic. This dummy mimics the interface and returns
    harmless values so no exceptions fire."""

    def clamp_min(self, _):
        import torch

        return torch.tensor(1.0)


def _run_one(cfg: argparse.Namespace) -> dict:
    r"""Run one training job under ``train_clean.train()`` with the model
    class and diagnostic helpers swapped per ``cfg.model_kind``. Returns
    the summary.json contents after the run completes."""
    from src.training import train_clean

    if cfg.model_kind == "hyp":
        # Default behavior. No patching needed.
        train_clean.train(cfg)

    elif cfg.model_kind == "euc":
        # Swap KettleGraphReasonerClean → EuclideanBaselineClean.
        from src.models.euclidean_baseline_clean import EuclideanBaselineClean

        original_cls = train_clean.KettleGraphReasonerClean
        try:
            train_clean.KettleGraphReasonerClean = EuclideanBaselineClean
            train_clean.train(cfg)
        finally:
            train_clean.KettleGraphReasonerClean = original_cls

    elif cfg.model_kind == "euc_plus":
        # Three patches:
        #   1. Swap the model class
        #   2. Patch distance_stats to not require node_logits
        #   3. Patch embedding_norm_stats to not require model.c
        # (3) is indirect — train_clean passes model.c to embedding_norm_stats.
        # We wrap the model after construction to give it a dummy .c property.
        from src.models.euclidean_plus_baseline import EuclideanPlusBaseline

        original_cls = train_clean.KettleGraphReasonerClean
        original_dist = train_clean.distance_stats
        original_emb = train_clean.embedding_norm_stats

        class _WrappedEucPlus(EuclideanPlusBaseline):
            r"""EuclideanPlusBaseline with a .c property so train_clean's
            ``embedding_norm_stats(h, model.c)`` call doesn't blow up."""

            @property
            def c(self):
                return _DummyC()

        try:
            train_clean.KettleGraphReasonerClean = _WrappedEucPlus
            train_clean.distance_stats = _patched_distance_stats
            train_clean.embedding_norm_stats = _patched_embedding_norm_stats
            train_clean.train(cfg)
        finally:
            train_clean.KettleGraphReasonerClean = original_cls
            train_clean.distance_stats = original_dist
            train_clean.embedding_norm_stats = original_emb

    else:
        raise ValueError(
            f"model_kind must be one of {_MODEL_KINDS}, got {cfg.model_kind!r}"
        )

    summary_path = Path(cfg.out) / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"train_clean did not produce {summary_path}. The run may have "
            "failed; check stdout above."
        )
    with open(summary_path, "r") as f:
        return json.load(f)


def _extract_final_ndcg(summary: dict, task: int) -> float:
    by_task = summary.get("final_val", {}).get("by_task_type", {})
    task_block = by_task.get(str(task)) or by_task.get(task) or {}
    val = task_block.get("ndcg@10")
    if val is None:
        raise RuntimeError(
            f"summary.json has no val nDCG@10 for task {task}. "
            f"Available keys under by_task_type: {list(by_task)}"
        )
    return float(val)


def _summarize(values: list[float]) -> dict:
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "min": values[0], "max": values[0]}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def _print_results_table(results: dict) -> None:
    print("\n" + "=" * 84)
    print("THREE-MODEL, THREE-SEED COMPARISON RESULTS")
    print("=" * 84)
    print(f"\nTask: {results['task']}")
    print(
        f"Epochs: {results['epochs']}  LR: {results['lr']}  "
        f"Hidden: {results['hidden_dim']}  Seeds: {results['seeds']}"
    )
    print()
    print(f"{'Model':<28} {'Mean nDCG@10':>14} {'Std':>8} " f"{'Min':>8} {'Max':>8}")
    print("-" * 72)
    for kind in results["models"]:
        s = results[f"{kind}_summary"]
        label: str = _MODEL_LABELS.get(kind) or str(kind)
        print(
            f"{label:<28} {s['mean']:>14.4f} {s['std']:>8.4f} "
            f"{s['min']:>8.4f} {s['max']:>8.4f}"
        )

    # Pairwise comparisons to the hyperbolic baseline (if present).
    print()
    if "hyp" in results["models"]:
        hyp_mean = results["hyp_summary"]["mean"]
        hyp_std = results["hyp_summary"]["std"]
        for other in results["models"]:
            if other == "hyp":
                continue
            other_mean = results[f"{other}_summary"]["mean"]
            other_std = results[f"{other}_summary"]["std"]
            diff = hyp_mean - other_mean
            pooled_std = max(hyp_std, other_std)
            label: str = _MODEL_LABELS.get(other) or str(other)
            print(f"Hyperbolic − {label}: {diff:+.4f}")
            if pooled_std == 0.0:
                print("  (Single seed; no noise estimate.)")
            elif abs(diff) < pooled_std:
                print("  Gap smaller than 1σ — within seed noise. No clear winner.")
            elif abs(diff) < 2 * pooled_std:
                print("  Gap between 1σ and 2σ — suggestive. Consider more seeds.")
            else:
                direction = "hyperbolic" if diff > 0 else label
                print(f"  Gap >2σ. {direction.capitalize()} wins with confidence.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(_MODEL_KINDS),
        choices=_MODEL_KINDS,
        help=f"Which model kinds to run. Default: all three. " f"Valid: {_MODEL_KINDS}",
    )
    parser.add_argument("--out-dir", type=str, default="runs/compare_v2")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If summary.json already exists for a (model, seed) pair, "
        "read it instead of re-training. Useful for resuming after a "
        "crash or adding a single model to an existing comparison.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ndcgs_by_model: dict[str, list[float]] = {k: [] for k in args.models}

    for seed in args.seeds:
        for model_kind in args.models:
            run_out = out_dir / f"{model_kind}_seed_{seed}"
            existing_summary = run_out / "summary.json"

            if args.skip_existing and existing_summary.exists():
                print(f"\n[skip-existing] reusing {existing_summary}")
                with open(existing_summary) as f:
                    summary = json.load(f)
            else:
                print(f"\n{'#' * 84}")
                print(f"# {_MODEL_LABELS.get(model_kind, model_kind)}  seed={seed}")
                print(f"{'#' * 84}")
                cfg = _build_cfg(
                    task=args.task,
                    corpus=args.corpus,
                    epochs=args.epochs,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    lr=args.lr,
                    out=str(run_out),
                    seed=seed,
                    cuda=args.cuda,
                    model_kind=model_kind,
                )
                summary = _run_one(cfg)

            ndcgs_by_model[model_kind].append(_extract_final_ndcg(summary, args.task))

    results = {
        "task": args.task,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "seeds": args.seeds,
        "models": args.models,
    }
    for kind in args.models:
        results[f"{kind}_ndcgs"] = ndcgs_by_model[kind]
        results[f"{kind}_summary"] = _summarize(ndcgs_by_model[kind])

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    _print_results_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
