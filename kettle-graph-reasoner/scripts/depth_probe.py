r"""Offline per-round linear probe — AttnRes Phase-1 diagnostic.

Given a trained KGR checkpoint, iterate the val split, collect the per-round
hyperbolic/Euclidean node embeddings h_r for each message-passing round r,
and fit a frozen-features logistic-regression probe per (task_type, round).
The round at which a task's probe peaks is where that task's discriminative
signal lives in depth — direct evidence for or against the AttnRes-style
dilution hypothesis (early rounds carry Task-1 signal but the final-round
representation is what the scoring head consumes).

Usage:
    py -m scripts.depth_probe \
        --checkpoint runs/my_run/best.pt \
        --corpus src/data/corpus/tier1 \
        --split val \
        --out runs/my_run/depth_probe.json

Outputs JSON keyed by task_type with per-round accuracy + AUC. Prints a
table. Read-only against the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.corpus_dataset import CorpusDataset, collate_single  # noqa: E402
from src.models.layers import poincare_ops as P  # noqa: E402
from src.training.train import build_model, forward_sample  # noqa: E402


def _logmap_if_hyperbolic(h: torch.Tensor, kind: str, c) -> np.ndarray:
    """Probes operate in a flat space. For hyperbolic embeddings this means
    logmap0 back to the tangent at origin, which is what the scoring head
    also consumes — so the probe sees the same view the downstream task
    does, differenced only by depth."""
    if kind == "hyperbolic":
        h = P.logmap0(h, c)
    return h.detach().cpu().numpy()


def _fit_and_score(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit a logistic regression with default regularization; return
    accuracy and ROC-AUC. Labels in [0, 1] are binarized at 0.5 because the
    provenance task has graded 1/d targets."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    y_bin = (y > 0.5).astype(np.int64)
    # Degenerate-label guard: a single class in this slice means probe is
    # undefined — report nan rather than crashing the whole sweep.
    if y_bin.sum() == 0 or y_bin.sum() == y_bin.size:
        return {"acc": float("nan"), "auc": float("nan"), "n": int(y_bin.size)}

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y_bin)
    score = clf.predict_proba(X)[:, 1]
    pred = (score >= 0.5).astype(np.int64)
    acc = float((pred == y_bin).mean())
    try:
        auc = float(roc_auc_score(y_bin, score))
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "auc": auc, "n": int(y_bin.size)}


def run(cfg: argparse.Namespace) -> None:
    device = torch.device("cuda" if (cfg.cuda and torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    train_cfg = ckpt.get("cfg", {})
    kind = train_cfg.get("model", cfg.model)
    hidden_dim = train_cfg.get("hidden_dim", cfg.hidden_dim)
    num_layers = train_cfg.get("num_layers", cfg.num_layers)
    hierarchy_subspace_dim = train_cfg.get("hierarchy_subspace_dim", 0)

    dataset = CorpusDataset(cfg.corpus, split=cfg.split, split_seed=train_cfg.get("seed", 0))
    print(f"[probe] kind={kind} L={num_layers} hidden={hidden_dim} {cfg.split}={len(dataset)}")

    model = build_model(
        kind,
        dataset,
        hidden_dim,
        num_layers,
        hierarchy_subspace_dim=hierarchy_subspace_dim,
        log_depth=True,  # force per-round stash regardless of how it was trained
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_single)

    # features[(task_type, round)] = list of (N, hidden_dim) numpy arrays;
    # labels[task_type] = list of (N,) numpy arrays. Stacked per-task at the
    # end. Per-task bucketing because label semantics (binary vs graded) and
    # positive-rate differ sharply across the 5 tasks.
    features: dict = defaultdict(lambda: defaultdict(list))
    labels: dict = defaultdict(list)

    c_val = getattr(model, "c", None) if kind == "hyperbolic" else None
    with torch.no_grad():
        for sample in loader:
            out = forward_sample(model, sample, device)
            if out.per_round_embeddings is None:
                raise RuntimeError(
                    "Model did not return per_round_embeddings; log_depth plumbing broken."
                )
            tt = int(sample.task_type)
            y = sample.labels.cpu().numpy()
            labels[tt].append(y)
            for r, h_r in enumerate(out.per_round_embeddings):
                features[tt][r].append(_logmap_if_hyperbolic(h_r, kind, c_val))

    results: dict = {}
    for tt in sorted(labels.keys()):
        y_all = np.concatenate(labels[tt], axis=0)
        results[str(tt)] = {}
        for r in sorted(features[tt].keys()):
            X_all = np.concatenate(features[tt][r], axis=0)
            stats = _fit_and_score(X_all, y_all)
            results[str(tt)][f"round_{r}"] = stats

    # Print a compact table.
    rounds = list(range(num_layers))
    header = "task | " + " | ".join(f"r{r:<5}" for r in rounds)
    print(header)
    print("-" * len(header))
    for tt in sorted(int(k) for k in results.keys()):
        row = results[str(tt)]
        cells = []
        for r in rounds:
            s = row.get(f"round_{r}", {})
            cells.append(f"{s.get('auc', float('nan')):.3f}")
        print(f"  {tt}  | " + " | ".join(f"{c:<6}" for c in cells))
    print("(cells = probe AUC per round; argmax per row = depth at which task signal peaks)")

    if cfg.out:
        os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
        with open(cfg.out, "w") as f:
            json.dump({"kind": kind, "num_layers": num_layers, "results": results}, f, indent=2)
        print(f"[probe] wrote {cfg.out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--out", type=str, default="")
    p.add_argument("--cuda", action="store_true")
    # Fallbacks if checkpoint lacks cfg dict (older runs).
    p.add_argument("--model", type=str, default="hyperbolic",
                   choices=["hyperbolic", "euclidean", "euclidean_plus"])
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
