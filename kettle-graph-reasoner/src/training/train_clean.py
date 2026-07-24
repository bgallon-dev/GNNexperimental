r"""Training entry point for the clean KGR model.

Single-task training with distance-based scoring. No aux depth head, no
radial regularizer, no multi-task loss composition, no concat_depth, no
subspace routing. The only thing this script does is:

  1. Load the corpus filtered to one task type.
  2. Build KettleGraphReasonerClean.
  3. Train with RiemannianAdam on task loss (MSE or BCE).
  4. Log |h| telemetry and per-epoch val metrics.

Run (one task at a time):

    py -m src.training.train_clean --task 0 --out runs/clean_task0
    py -m src.training.train_clean --task 2 --out runs/clean_task2
    py -m src.training.train_clean --task 3 --out runs/clean_task3

If these three runs each climb on their own task, the architecture works.
If any fails, you know something specific about that task-architecture
interaction. Either way, you get interpretable results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..data.corpus_dataset import CorpusDataset, collate_single
from ..models.hyperbolic_gnn_clean import KettleGraphReasonerClean
from ..models.layers import poincare_ops as P
from .metrics import MetricAccumulator


# Task-to-loss mapping mirrors the original loss.py. Task 1 (entity
# resolution) intentionally not supported here — it needs a different
# loss formulation (contrastive or margin-ranking) and is out of scope
# for the clean-scoring ablation.
BCE_TASKS = {0, 4, 5}
MSE_TASKS = {2, 3}
SUPPORTED_TASKS = BCE_TASKS | MSE_TASKS


def pick_loss_fn(task: int):
    if task in BCE_TASKS:
        return F.binary_cross_entropy
    if task in MSE_TASKS:
        return F.mse_loss
    raise ValueError(
        f"task {task} not supported by train_clean. Supported: {sorted(SUPPORTED_TASKS)}"
    )


def task_loss(
    output,
    node_labels: torch.Tensor,
    edge_index: torch.Tensor,
    task: int,
    edge_weight: float = 0.5,
    eps: float = 1e-6,
) -> dict:
    """Simple node + edge loss. No per-task reweighting (task 1 is not
    supported here, so the pos_weight code path is gone)."""
    loss_fn = pick_loss_fn(task)

    node_labels = node_labels.clamp(0.0, 1.0)
    node_pred = output.node_scores.clamp(eps, 1.0 - eps)
    node_l = loss_fn(node_pred, node_labels)

    src, dst = edge_index[0], edge_index[1]
    edge_labels = 0.5 * (
        node_labels.index_select(0, src) + node_labels.index_select(0, dst)
    )
    edge_pred = output.edge_scores.clamp(eps, 1.0 - eps)
    edge_l = loss_fn(edge_pred, edge_labels)

    total = node_l + edge_weight * edge_l
    return {"loss": total, "node_loss": node_l.detach(), "edge_loss": edge_l.detach()}


def forward_sample(model: nn.Module, sample, device: torch.device):
    return model(
        node_features=sample.x.to(device),
        edge_index=sample.edge_index.to(device),
        edge_type=sample.edge_type.to(device),
        edge_descriptor=sample.edge_descriptor.to(device),
        query=sample.query.to(device),
        node_descriptor=sample.node_descriptor.to(device),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, task: int) -> dict:
    model.eval()
    acc = MetricAccumulator()
    total_loss = 0.0
    n = 0
    for sample in loader:
        out = forward_sample(model, sample, device)
        labels = sample.labels.to(device)
        loss = task_loss(out, labels, sample.edge_index.to(device), task)
        total_loss += float(loss["loss"])
        n += 1
        acc.add(out.node_scores.cpu(), labels.cpu(), task)
    model.train()
    summary = acc.summary()
    summary["val_loss"] = total_loss / max(n, 1)
    return summary


def embedding_norm_stats(h: torch.Tensor, c: Optional[torch.Tensor]) -> dict:
    h = h.detach()
    norms = h.norm(dim=-1)
    stats = {
        "mean_norm": float(norms.mean()),
        "max_norm": float(norms.max()),
        "min_norm": float(norms.min()),
        "std_norm": float(norms.std(unbiased=False)) if norms.numel() > 1 else 0.0,
    }
    if c is not None:
        stats["boundary"] = 1.0 / float(c.clamp_min(P.MIN_NORM).sqrt())
    return stats


def distance_stats(out) -> dict:
    """How separated are nodes in hyperbolic distance from the query?
    If all nodes are at the same distance from h_q, the scoring function
    can't rank them. Std of distances is the clean diagnostic."""
    d = (-out.node_logits).detach()  # distances are the negated logits
    return {
        "dist_mean": float(d.mean()),
        "dist_std": float(d.std(unbiased=False)) if d.numel() > 1 else 0.0,
        "dist_min": float(d.min()),
        "dist_max": float(d.max()),
    }


def train(cfg: argparse.Namespace) -> None:
    if cfg.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"--task {cfg.task} not supported. Valid: {sorted(SUPPORTED_TASKS)}"
        )

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if (cfg.cuda and torch.cuda.is_available()) else "cpu")

    # Filter both splits to the single task.
    train_set = CorpusDataset(
        cfg.corpus, split="train", split_seed=cfg.seed, include_tasks=[cfg.task]
    )
    val_set = CorpusDataset(
        cfg.corpus, split="val", split_seed=cfg.seed, include_tasks=[cfg.task]
    )
    print(f"[data] task={cfg.task}  train={len(train_set)}  val={len(val_set)}")

    model = KettleGraphReasonerClean(
        node_feat_dim=train_set.node_feat_dim,
        edge_feat_dim=train_set.edge_feat_dim_schema,
        query_dim=train_set.query_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_edge_types_max=train_set.num_edge_types_max,
        node_feat_dim_schema=train_set.node_feat_dim_schema,
        tangent_scale_init=cfg.tangent_scale,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] clean  hidden={cfg.hidden_dim}  L={cfg.num_layers}  params={n_params:,}")

    # RiemannianAdam for the hyperbolic parameters. Same as original.
    try:
        from geoopt.optim import RiemannianAdam
        opt = RiemannianAdam(model.parameters(), lr=cfg.lr)
    except ImportError:
        raise RuntimeError("geoopt is required. `pip install geoopt`.")

    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
    scheduler: Optional[object] = None
    if cfg.lr_schedule == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    elif cfg.lr_schedule == "step":
        scheduler = StepLR(opt, step_size=cfg.lr_step_epoch, gamma=0.1)

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, collate_fn=collate_single,
        num_workers=2, prefetch_factor=2, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, collate_fn=collate_single,
        num_workers=2, prefetch_factor=2, persistent_workers=True,
    )

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    epochs_without_improvement = 0

    step = 0
    ema_loss: Optional[float] = None
    final_train_summary: dict = {}
    final_val_summary: dict = {}

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_acc = MetricAccumulator()

        for sample in train_loader:
            out = forward_sample(model, sample, device)
            labels = sample.labels.to(device)
            loss_dict = task_loss(
                out, labels, sample.edge_index.to(device), cfg.task,
                edge_weight=cfg.edge_loss_weight,
            )
            loss = loss_dict["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()

            train_acc.add(out.node_scores.detach().cpu(), labels.cpu(), cfg.task)
            ema_loss = float(loss) if ema_loss is None else 0.98 * ema_loss + 0.02 * float(loss)

            if step % cfg.log_every == 0:
                nstats = embedding_norm_stats(out.node_embeddings, getattr(model, "c", None))
                dstats = distance_stats(out)
                q_point = getattr(out, "query_point", None)
                q_norm = float(q_point.detach().norm()) if q_point is not None else float("nan")
                print(
                    f"[train] epoch={epoch} step={step} "
                    f"loss={float(loss):.4f} ema={ema_loss:.4f} "
                    f"node={float(loss_dict['node_loss']):.4f} "
                    f"edge={float(loss_dict['edge_loss']):.4f} "
                    f"|h|_mean={nstats['mean_norm']:.3f} "
                    f"|h|_max={nstats['max_norm']:.3f} "
                    f"|h|_std={nstats['std_norm']:.4f} "
                    f"|h_q|={q_norm:.3f} "
                    f"d_mean={dstats['dist_mean']:.3f} "
                    f"d_std={dstats['dist_std']:.4f}"
                )
            step += 1

        train_summary = train_acc.summary()
        val = evaluate(model, val_loader, device, cfg.task)
        dt = time.time() - t0
        print(
            f"[val]   epoch={epoch} val_loss={val['val_loss']:.4f} "
            f"P@10={val['overall']['p@10']:.3f} "
            f"R@10={val['overall']['r@10']:.3f} "
            f"nDCG@10={val['overall']['ndcg@10']:.3f} "
            f"(epoch {dt:.1f}s)"
        )
        tr_ndcg = train_summary["by_task_type"].get(cfg.task, {}).get("ndcg@10", float("nan"))
        va_ndcg = val["by_task_type"].get(cfg.task, {}).get("ndcg@10", float("nan"))
        print(f"[gap]   epoch={epoch} task={cfg.task} train_nDCG={tr_ndcg:.3f} val_nDCG={va_ndcg:.3f} gap={tr_ndcg - va_ndcg:+.3f}")

        (out_dir / f"val_epoch_{epoch}.json").write_text(json.dumps(val, indent=2))
        (out_dir / f"train_epoch_{epoch}.json").write_text(json.dumps(train_summary, indent=2))

        final_train_summary = train_summary
        final_val_summary = val

        if val["val_loss"] < best_val:
            best_val = val["val_loss"]
            epochs_without_improvement = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "cfg": vars(cfg), "val": val},
                out_dir / "best.pt",
            )
            print(f"[ckpt]  saved best @ epoch={epoch} val_loss={best_val:.4f}")
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        if cfg.early_stop_patience > 0 and epochs_without_improvement >= cfg.early_stop_patience:
            print(f"[early_stop] no val_loss improvement for {cfg.early_stop_patience} epochs — stopping at epoch {epoch}.")
            break

    (out_dir / "summary.json").write_text(json.dumps({
        "task": cfg.task,
        "n_params": n_params,
        "epochs": cfg.epochs,
        "final_train": final_train_summary,
        "final_val": final_val_summary,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True, help="single task id to train on")
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-schedule", choices=["none", "cosine", "step"], default="none")
    p.add_argument("--lr-step-epoch", type=int, default=15)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--edge-loss-weight", type=float, default=0.5)
    p.add_argument("--tangent-scale", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", type=str, default="runs/clean_default")
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
