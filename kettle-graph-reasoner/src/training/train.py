r"""Training entry point for KGR.

One (graph, task) pair per step. The hyperbolic model is trained with
``geoopt.optim.RiemannianAdam`` (handles Euclidean + manifold params
together); the Euclidean baseline uses plain ``torch.optim.Adam``. The
same loop drives both so the A/B comparison stays honest.

Run:

    py -m src.training.train --model hyperbolic --epochs 20 --out runs/hyp_tier1
    py -m src.training.train --model euclidean  --epochs 20 --out runs/euc_tier1

If hyperbolic training is unstable (NaN loss, ||h||→1/√c saturating),
drop ``--lr`` to 3e-4 before touching anything else — curvature
amplifies effective step size near the Poincaré ball boundary.
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


# ---------------------------------------------------------------------------
# Terminal color helpers — green = what we want to see, yellow = borderline,
# red = pathological. Auto-disabled if stdout isn't a TTY or --no-color is set.
# ---------------------------------------------------------------------------
_ANSI = {
    "good":    "\033[32m",   # green
    "warn":    "\033[33m",   # yellow
    "bad":     "\033[31m",   # red
    "neutral": "",
    "reset":   "\033[0m",
}
_COLOR_ENABLED = False


def _enable_color(enable: bool) -> None:
    global _COLOR_ENABLED
    if not enable:
        _COLOR_ENABLED = False
        return
    if not sys.stdout.isatty():
        _COLOR_ENABLED = False
        return
    if sys.platform == "win32":
        # Kicks the console into VT-processing mode on Win10+; harmless elsewhere.
        os.system("")
    _COLOR_ENABLED = True


def _c(text: str, status: str) -> str:
    if not _COLOR_ENABLED or status == "neutral":
        return text
    return f"{_ANSI[status]}{text}{_ANSI['reset']}"


def _status_hmax(v: float) -> str:
    # Healthy: max spreads toward 0.8–0.9. Saturation at ~0.99 is pathological.
    if v >= 0.99:
        return "bad"
    if v >= 0.95:
        return "warn"
    return "good"


def _status_hmean(v: float) -> str:
    # Healthy range 0.2–0.6 per Known Issues. Outside that, warn; far outside, bad.
    if 0.2 <= v <= 0.6:
        return "good"
    if 0.1 <= v <= 0.8:
        return "warn"
    return "bad"


def _status_hstd(v: float) -> str:
    # Per-node radial differentiation should be non-trivial. Shell-collapse → std→0.
    if v >= 0.05:
        return "good"
    if v >= 0.01:
        return "warn"
    return "bad"


def _status_ndcg_task0(v: float) -> str:
    # Run A target: stabilize above 0.340 (vs. prior oscillation).
    if v >= 0.34:
        return "good"
    if v >= 0.30:
        return "warn"
    return "bad"


def _status_ndcg_generic(v: float) -> str:
    if v >= 0.50:
        return "good"
    if v >= 0.30:
        return "warn"
    return "bad"


def _status_gap(gap: float) -> str:
    # Positive small gap = healthy generalization. Large positive = overfit.
    # Negative (val > train) was the pathology the subspace is meant to fix.
    if -0.02 <= gap <= 0.05:
        return "good"
    if -0.05 <= gap <= 0.10:
        return "warn"
    return "bad"


def _status_ratio(v: float) -> str:
    # rad/tan > ~0.1 means the radial axis is carrying signal.
    if v >= 0.10:
        return "good"
    if v >= 0.01:
        return "warn"
    return "bad"

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ..data.corpus_dataset import CorpusDataset, collate_single
from ..models.hyperbolic_gnn import KettleGraphReasoner
from ..models.euclidean_baseline import EuclideanBaseline
from ..models.euclidean_plus_baseline import EuclideanPlusBaseline
from ..models.layers import poincare_ops as P
from .loss import relevance_loss
from .metrics import MetricAccumulator

# Layer one-hot sits at x[:, LAYER_SLICE] (source=0, claim=1, entity=2, auxiliary=3).
# See src/data/feature_encoder.py. Used as the auxiliary depth target to force
# radial differentiation in the hyperbolic embedding: the aux head sees only
# ||h|| and must predict layer, so gradients back-propagate to node-radial-
# position. Without it the scoring head discriminates from direction alone and
# the radial dimension stays unused.
LAYER_SLICE = slice(12, 16)
NUM_LAYERS_DEPTH = 4


class DepthHead(nn.Module):
    """Maps a single scalar ||h|| per node to layer-class logits. Kept tiny on
    purpose: the point is to create radial gradient pressure, not to do clever
    depth inference.

    When ``k > 0``, the head reads the radius of the logmap0-tangent slice
    ``logmap0(h)[:, :k]`` — the same k-dim subspace the hierarchy scoring head
    consumes. This concentrates aux-depth gradient pressure in the reserved
    subspace and leaves the remaining hidden_dim − k coordinates free for the
    proximity tasks, rather than pulling the full embedding's radial norm.
    """

    def __init__(
        self,
        num_classes: int = NUM_LAYERS_DEPTH,
        hidden: int = 8,
        k: int = 0,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(
        self,
        h: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.k > 0:
            if c is None:
                raise ValueError("DepthHead with k>0 requires curvature c")
            h_slice = P.logmap0(h, c)[..., : self.k]
        else:
            h_slice = h
        r = h_slice.norm(dim=-1, keepdim=True)  # (N, 1)
        return self.net(r)


def build_model(
    kind: str,
    dataset: CorpusDataset,
    hidden_dim: int,
    num_layers: int,
    hierarchy_subspace_dim: int = 0,
    log_depth: bool = False,
    concat_depth: bool = False,
) -> nn.Module:
    if kind == "hyperbolic":
        return KettleGraphReasoner(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            query_dim=dataset.query_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types_max=dataset.num_edge_types_max,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
            hierarchy_subspace_dim=hierarchy_subspace_dim,
            log_depth=log_depth,
            concat_depth=concat_depth,
        )
    if kind == "euclidean_plus":
        return EuclideanPlusBaseline(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            query_dim=dataset.query_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types_max=dataset.num_edge_types_max,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
            log_depth=log_depth,
        )
    if kind == "euclidean":
        return EuclideanBaseline(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            query_dim=dataset.query_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            log_depth=log_depth,
        )
    raise ValueError(f"unknown model kind: {kind!r}")


def build_optimizer_for_params(params, kind: str, lr: float):
    if kind == "hyperbolic":
        try:
            from geoopt.optim import RiemannianAdam
        except ImportError as e:
            raise RuntimeError(
                "geoopt is required for hyperbolic training. `pip install geoopt`."
            ) from e
        return RiemannianAdam(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


def forward_sample(model: nn.Module, sample, device: torch.device):
    kwargs = dict(
        node_features=sample.x.to(device),
        edge_index=sample.edge_index.to(device),
        edge_type=sample.edge_type.to(device),
        edge_descriptor=sample.edge_descriptor.to(device),
        query=sample.query.to(device),
        node_descriptor=sample.node_descriptor.to(device),
    )
    if isinstance(model, KettleGraphReasoner) and model.hierarchy_subspace_dim > 0:
        kwargs["task_type"] = sample.task_type
    return model(**kwargs)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    acc = MetricAccumulator()
    total_loss = 0.0
    n = 0
    for sample in loader:
        out = forward_sample(model, sample, device)
        labels = sample.labels.to(device)
        loss = relevance_loss(
            out, labels, sample.edge_index.to(device), sample.task_type
        )
        total_loss += float(loss["loss"])
        n += 1
        acc.add(out.node_scores.cpu(), labels.cpu(), sample.task_type)
    model.train()
    summary = acc.summary()
    summary["val_loss"] = total_loss / max(n, 1)
    return summary


def embedding_norm_stats(out, kind: str, c: Optional[torch.Tensor]) -> dict:
    h = out.node_embeddings.detach()
    norms = h.norm(dim=-1)
    stats = {
        "mean_norm": float(norms.mean()),
        "max_norm": float(norms.max()),
        "min_norm": float(norms.min()),
        "std_norm": float(norms.std(unbiased=False)) if norms.numel() > 1 else 0.0,
    }
    if kind == "hyperbolic" and c is not None:
        stats["boundary"] = 1.0 / float(c.clamp_min(P.MIN_NORM).sqrt())
    return stats


def _parse_task_weights(spec: str) -> dict:
    """Parse 'w0,w1,w2,w3,w4' or '' -> {0:w0,...}. Missing entries default 1."""
    weights = {i: 1.0 for i in range(5)}
    spec = (spec or "").strip()
    if not spec:
        return weights
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for i, p in enumerate(parts):
        if i >= 5:
            break
        weights[i] = float(p)
    return weights


def _radial_tangent_grad(node_loss: torch.Tensor, h: torch.Tensor) -> dict:
    """Decompose d node_loss / d h into radial and tangential components.
    Radial direction is h/||h||; radial grad magnitude tells us how hard the
    task is pushing on ||h||, the hyperbolic hierarchy axis."""
    g = torch.autograd.grad(node_loss, h, retain_graph=True, create_graph=False)[0]
    h_det = h.detach()
    h_norm = h_det.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    radial_dir = h_det / h_norm
    radial = (g * radial_dir).sum(dim=-1)
    tangent = g - radial.unsqueeze(-1) * radial_dir
    return {
        "grad_radial": float(radial.norm()),
        "grad_tangent": float(tangent.norm()),
        "grad_total": float(g.norm()),
    }


def per_round_norm_stats(per_round, kind: str, c: Optional[torch.Tensor]) -> list:
    """Per-round analogue of embedding_norm_stats. Returns a list indexed by
    round, each entry the same dict shape embedding_norm_stats produces."""
    if per_round is None:
        return []
    out = []
    for h in per_round:
        h_det = h.detach()
        norms = h_det.norm(dim=-1)
        stat = {
            "mean_norm": float(norms.mean()),
            "max_norm": float(norms.max()),
            "min_norm": float(norms.min()),
            "std_norm": float(norms.std(unbiased=False)) if norms.numel() > 1 else 0.0,
        }
        if kind == "hyperbolic" and c is not None:
            stat["boundary"] = 1.0 / float(c.clamp_min(P.MIN_NORM).sqrt())
        out.append(stat)
    return out


def _per_round_grad_norms(loss: torch.Tensor, per_round) -> list:
    """||d loss / d h_r|| for each stashed per-round embedding h_r. Tests
    whether gradient reaches early rounds or concentrates in the final one —
    the AttnRes diagnostic signature for depth-wise signal imbalance."""
    if per_round is None or len(per_round) == 0:
        return []
    grads = torch.autograd.grad(
        loss, list(per_round), retain_graph=True, allow_unused=True
    )
    norms = []
    for g in grads:
        norms.append(float(g.norm()) if g is not None else 0.0)
    return norms


def _bucket_grad_norms(model: nn.Module) -> dict:
    # manifold: feeds / lives on the Poincaré ball. edge_attn: edge-typed
    # attention + schema encoder. head: Euclidean scoring MLPs + query
    # projection. Diagnostic rule: head >> manifold with flat val = head
    # bottleneck; manifold vanishing across depth = geometric signal flow.
    buckets = {"manifold": 0.0, "edge_attn": 0.0, "head": 0.0, "other": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g2 = float(p.grad.detach().pow(2).sum())
        if name.startswith(("node_in", "tangent_scale", "_c", "mp_layers")):
            buckets["manifold"] += g2
        elif name.startswith(("attn_layers", "schema_encoder")):
            buckets["edge_attn"] += g2
        elif name.startswith(("node_score", "edge_score", "query_in")):
            buckets["head"] += g2
        else:
            buckets["other"] += g2
    return {k: v ** 0.5 for k, v in buckets.items()}


def train(cfg: argparse.Namespace) -> None:
    _enable_color(enable=not getattr(cfg, "no_color", False))
    if cfg.radial_reg_weight_end is None:
        cfg.radial_reg_weight_end = cfg.radial_reg_weight
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if (cfg.cuda and torch.cuda.is_available()) else "cpu")

    train_set_full = CorpusDataset(cfg.corpus, split="train", split_seed=cfg.seed)
    val_set = CorpusDataset(cfg.corpus, split="val", split_seed=cfg.seed)
    if cfg.train_frac < 1.0:
        g = torch.Generator().manual_seed(cfg.seed)
        perm = torch.randperm(len(train_set_full), generator=g).tolist()
        n_keep = max(1, int(cfg.train_frac * len(train_set_full)))
        train_set = Subset(train_set_full, perm[:n_keep])
        print(f"[data] train_frac={cfg.train_frac:.2f} -> {n_keep}/{len(train_set_full)} tasks")
    else:
        train_set = train_set_full
    print(f"[data] train: {len(train_set)} tasks over {len(train_set_full.files)} graphs")
    print(f"[data] val:   {len(val_set)} tasks over {len(val_set.files)} graphs")

    model = build_model(
        cfg.model,
        train_set_full,
        cfg.hidden_dim,
        cfg.num_layers,
        hierarchy_subspace_dim=cfg.hierarchy_subspace_dim,
        log_depth=cfg.log_depth_diagnostics,
        concat_depth=getattr(cfg, "concat_depth", False),
    ).to(device)
    # Always read k back from the constructed model (not from cfg) so any
    # plumbing bug between CLI/config and the constructor surfaces here
    # instead of silently defaulting to 0.
    if isinstance(model, KettleGraphReasoner):
        actual_k = model.hierarchy_subspace_dim
        assert actual_k == cfg.hierarchy_subspace_dim, (
            f"hierarchy_subspace_dim mismatch: cfg={cfg.hierarchy_subspace_dim} "
            f"but model={actual_k} — check build_model plumbing"
        )
        if actual_k > 0:
            print(
                f"[model] hierarchy_subspace_dim={actual_k}: "
                f"Task 0 uses first {actual_k} dims, "
                f"Tasks 1-4 use remaining {cfg.hidden_dim - actual_k}"
            )
        else:
            print("[model] hierarchy_subspace_dim=0 (shared head for all tasks)")

    task_weights = _parse_task_weights(cfg.task_loss_weights)
    if any(w != 1.0 for w in task_weights.values()):
        print(f"[loss]  per-task weights: {task_weights}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {cfg.model}  hidden={cfg.hidden_dim}  L={cfg.num_layers}  params={n_params:,}")

    depth_head: Optional[DepthHead] = None
    if cfg.model in ("hyperbolic", "euclidean_plus") and cfg.aux_depth_weight > 0:
        aux_k = (
            model.hierarchy_subspace_dim
            if isinstance(model, KettleGraphReasoner)
            else 0
        )
        depth_head = DepthHead(k=aux_k).to(device)
        print(f"[aux]   depth head active, weight={cfg.aux_depth_weight}")

    params = list(model.parameters())
    if depth_head is not None:
        params += list(depth_head.parameters())
    opt = build_optimizer_for_params(params, cfg.model, cfg.lr)

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
    )

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf

    step = 0
    log_every = cfg.log_every
    ema_loss = None
    last_norm_stats: dict = {}

    final_train_summary: dict = {}
    final_val_summary: dict = {}
    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_acc = MetricAccumulator()
        per_task_grad_accum: dict = {}
        # depth_diag[task] = {"round_norms": [sum over steps of ||h_r||_mean],
        #                     "round_grads": [sum over steps of ||dL/dh_r||],
        #                     "n": step count}
        depth_diag_accum: dict = {}
        # Linear decay of radial reg: full strength early to block boundary
        # saturation while the model is still finding its scale, then fade so
        # later epochs can use more of the ball's volume.
        if cfg.epochs > 1:
            frac = epoch / (cfg.epochs - 1)
        else:
            frac = 0.0
        radial_w = cfg.radial_reg_weight + frac * (
            cfg.radial_reg_weight_end - cfg.radial_reg_weight
        )
        for sample in train_loader:
            if cfg.limit_batches and step >= cfg.limit_batches:
                break
            out = forward_sample(model, sample, device)
            labels = sample.labels.to(device)
            loss_dict = relevance_loss(
                out,
                labels,
                sample.edge_index.to(device),
                sample.task_type,
                edge_weight=cfg.edge_loss_weight,
                task_weight=task_weights.get(sample.task_type, 1.0),
            )
            loss = loss_dict["loss"]

            # Per-task radial/tangent grad diag: how much of d node_loss / d h
            # is pushing along the ||h|| axis vs along the tangent. If Task 3
            # has an order-of-magnitude larger radial grad than Task 0 at
            # baseline, that is independent evidence for the flattening
            # mechanism. Requires --grad-diag (still costs one autograd pass).
            radial_grad_stats: Optional[dict] = None
            if cfg.grad_diag:
                radial_grad_stats = _radial_tangent_grad(
                    loss_dict["node_loss"], out.node_embeddings
                )
                tt = sample.task_type
                per_task_grad_accum.setdefault(
                    tt, {"radial": 0.0, "tangent": 0.0, "total": 0.0, "n": 0}
                )
                b = per_task_grad_accum[tt]
                b["radial"] += radial_grad_stats["grad_radial"]
                b["tangent"] += radial_grad_stats["grad_tangent"]
                b["total"] += radial_grad_stats["grad_total"]
                b["n"] += 1

            # Radial regularizer: applied only to the hyperbolic model's
            # final-layer node embeddings. Penalizes drift toward the ball
            # boundary without forbidding it — a counterforce against the
            # saturation attractor in the HNN stack. Intermediate layers are
            # not regularized; only what the scoring head sees.
            if cfg.model in ("hyperbolic", "euclidean_plus") and radial_w > 0:
                radial = out.node_embeddings.pow(2).sum(dim=-1).mean()
                loss = loss + radial_w * radial

            # Auxiliary depth loss: head sees only ||h|| and predicts the
            # node's hierarchy layer. Forces the model to encode depth
            # radially — without it, the scoring head discriminates from
            # direction alone and radial variance stays zero.
            aux_loss_val = 0.0
            if depth_head is not None:
                layer_target = (
                    sample.x[:, LAYER_SLICE].argmax(dim=-1).to(device)
                )
                c_arg = model.c if isinstance(model, KettleGraphReasoner) else None
                depth_logits = depth_head(out.node_embeddings, c_arg)
                aux_loss = torch.nn.functional.cross_entropy(
                    depth_logits, layer_target
                )
                loss = loss + cfg.aux_depth_weight * aux_loss
                aux_loss_val = float(aux_loss.detach())

            # Per-round diagnostic: compute activation-gradient norms BEFORE
            # the backward pass consumes the graph. Guarded by flag and by
            # presence of per_round_embeddings on the model output — no-op if
            # the model has not been extended to stash per-round h yet.
            per_round = getattr(out, "per_round_embeddings", None)
            if cfg.log_depth_diagnostics and per_round:
                c_val = getattr(model, "c", None) if cfg.model == "hyperbolic" else None
                round_norm_list = per_round_norm_stats(per_round, cfg.model, c_val)
                round_grad_list = _per_round_grad_norms(
                    loss_dict["node_loss"], per_round
                )
                tt = sample.task_type
                bucket = depth_diag_accum.setdefault(
                    tt,
                    {
                        "round_norms": [0.0] * len(round_norm_list),
                        "round_grads": [0.0] * len(round_grad_list),
                        "n": 0,
                    },
                )
                for i, s in enumerate(round_norm_list):
                    bucket["round_norms"][i] += s["mean_norm"]
                for i, g in enumerate(round_grad_list):
                    bucket["round_grads"][i] += g
                bucket["n"] += 1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms = _bucket_grad_norms(model) if cfg.grad_diag else None
            torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
            opt.step()

            train_acc.add(
                out.node_scores.detach().cpu(), labels.cpu(), sample.task_type
            )

            ema_loss = float(loss) if ema_loss is None else 0.98 * ema_loss + 0.02 * float(loss)
            if step % log_every == 0:
                c_val = getattr(model, "c", None) if cfg.model == "hyperbolic" else None
                last_norm_stats = embedding_norm_stats(out, cfg.model, c_val)
                aux_str = f" aux={aux_loss_val:.4f}" if depth_head is not None else ""
                reg_str = f" reg={radial_w:.4f}" if cfg.model in ("hyperbolic", "euclidean_plus") else ""
                grad_str = (
                    f" g_mani={grad_norms['manifold']:.3f}"
                    f" g_attn={grad_norms['edge_attn']:.3f}"
                    f" g_head={grad_norms['head']:.3f}"
                    if grad_norms is not None
                    else ""
                )
                if radial_grad_stats is not None:
                    grad_str += (
                        f" t={sample.task_type}"
                        f" g_rad={radial_grad_stats['grad_radial']:.4f}"
                        f" g_tan={radial_grad_stats['grad_tangent']:.4f}"
                    )
                hmean = last_norm_stats["mean_norm"]
                hmax = last_norm_stats["max_norm"]
                hstd = last_norm_stats["std_norm"]
                print(
                    f"[train] epoch={epoch} step={step} "
                    f"loss={float(loss):.4f} ema={ema_loss:.4f} "
                    f"node={float(loss_dict['node_loss']):.4f} "
                    f"edge={float(loss_dict['edge_loss']):.4f}"
                    f"{aux_str}{reg_str} "
                    f"|h|_mean={_c(f'{hmean:.3f}', _status_hmean(hmean))} "
                    f"|h|_max={_c(f'{hmax:.3f}', _status_hmax(hmax))} "
                    f"|h|_min={last_norm_stats['min_norm']:.3f} "
                    f"|h|_std={_c(f'{hstd:.4f}', _status_hstd(hstd))}"
                    f"{grad_str}"
                )
            step += 1

        if cfg.limit_batches and step >= cfg.limit_batches:
            print(f"[train] reached limit_batches={cfg.limit_batches}, stopping.")
            break

        train_summary = train_acc.summary()
        val = evaluate(model, val_loader, device)
        dt = time.time() - t0
        overall_ndcg = val["overall"]["ndcg@10"]
        print(
            f"[val]   epoch={epoch} val_loss={val['val_loss']:.4f} "
            f"P@10={val['overall']['p@10']:.3f} "
            f"R@10={val['overall']['r@10']:.3f} "
            f"nDCG@10={_c(f'{overall_ndcg:.3f}', _status_ndcg_generic(overall_ndcg))} "
            f"(epoch {dt:.1f}s)"
        )
        for t, m in val["by_task_type"].items():
            ndcg = m["ndcg@10"]
            status = _status_ndcg_task0(ndcg) if int(t) == 0 else _status_ndcg_generic(ndcg)
            print(
                f"        task_type={t} P@10={m['p@10']:.3f} "
                f"R@10={m['r@10']:.3f} "
                f"nDCG@10={_c(f'{ndcg:.3f}', status)}"
            )

        if cfg.grad_diag and per_task_grad_accum:
            print(f"[grad]  epoch={epoch} per-task mean node_loss grad on h:")
            for t in sorted(per_task_grad_accum.keys()):
                b = per_task_grad_accum[t]
                n = max(b["n"], 1)
                ratio = (b["radial"] / n) / max(b["tangent"] / n, 1e-8)
                print(
                    f"        task_type={t} n={b['n']} "
                    f"g_radial={b['radial']/n:.4f} "
                    f"g_tangent={b['tangent']/n:.4f} "
                    f"ratio_rad/tan={_c(f'{ratio:.3f}', _status_ratio(ratio))}"
                )

        train_by_t = train_summary.get("by_task_type", {})
        for t, m in val["by_task_type"].items():
            tr = train_by_t.get(t, {}).get("ndcg@10", float("nan"))
            va = m["ndcg@10"]
            gap = tr - va
            print(
                f"[gap]   epoch={epoch} task_type={t} "
                f"train_nDCG={tr:.3f} val_nDCG={va:.3f} "
                f"gap={_c(f'{gap:+.3f}', _status_gap(gap))}"
            )

        depth_diag_summary: dict = {}
        if cfg.log_depth_diagnostics and depth_diag_accum:
            print(f"[depth] epoch={epoch} per-task per-round ||h||_mean and ||dL/dh_r||:")
            for t in sorted(depth_diag_accum.keys()):
                b = depth_diag_accum[t]
                n = max(b["n"], 1)
                norms_mean = [x / n for x in b["round_norms"]]
                grads_mean = [x / n for x in b["round_grads"]]
                depth_diag_summary[str(t)] = {
                    "n": b["n"],
                    "round_norms_mean": norms_mean,
                    "round_grads_mean": grads_mean,
                }
                norms_str = " ".join(f"{v:.3f}" for v in norms_mean)
                grads_str = " ".join(f"{v:.4f}" for v in grads_mean)
                print(
                    f"        task_type={t} n={b['n']} "
                    f"|h|_per_round=[{norms_str}] "
                    f"|dL/dh|_per_round=[{grads_str}]"
                )

        (out_dir / f"val_epoch_{epoch}.json").write_text(json.dumps(val, indent=2))
        (out_dir / f"train_epoch_{epoch}.json").write_text(json.dumps(train_summary, indent=2))
        if depth_diag_summary:
            (out_dir / f"depth_diag_epoch_{epoch}.json").write_text(
                json.dumps(depth_diag_summary, indent=2)
            )
        final_train_summary = train_summary
        final_val_summary = val

        if val["val_loss"] < best_val:
            best_val = val["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "cfg": vars(cfg),
                    "val": val,
                },
                out_dir / "best.pt",
            )
            print(_c(f"[ckpt]  saved best @ epoch={epoch} val_loss={best_val:.4f}", "good"))

    train_by_t = final_train_summary.get("by_task_type", {})
    val_by_t = final_val_summary.get("by_task_type", {})
    gap_by_task = {
        str(t): {
            "train_ndcg@10": train_by_t.get(t, {}).get("ndcg@10", float("nan")),
            "val_ndcg@10": m.get("ndcg@10", float("nan")),
            "gap": train_by_t.get(t, {}).get("ndcg@10", float("nan"))
            - m.get("ndcg@10", float("nan")),
        }
        for t, m in val_by_t.items()
    }
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": cfg.model,
                "train_frac": cfg.train_frac,
                "n_params": n_params,
                "epochs": cfg.epochs,
                "final_train": final_train_summary,
                "final_val": final_val_summary,
                "gap_by_task": gap_by_task,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=["hyperbolic", "euclidean", "euclidean_plus"],
        required=True,
    )
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--edge-loss-weight", type=float, default=0.5)
    p.add_argument(
        "--radial-reg-weight",
        type=float,
        default=0.1,
        help="Coefficient on mean ||h||^2 penalty for hyperbolic embeddings. "
        "Counterforce against the Poincaré boundary attractor. "
        "Sweep {0.001, 0.01, 0.1}; ignored for the Euclidean baseline. "
        "Linearly decays to --radial-reg-weight-end across epochs.",
    )
    p.add_argument(
        "--radial-reg-weight-end",
        type=float,
        default=None,
        help="Final radial reg weight at the last epoch; defaults to the "
        "start value (no decay). Use e.g. 0.001 with start=0.1 to fade the "
        "regularizer so later epochs can use more of the ball's volume.",
    )
    p.add_argument(
        "--aux-depth-weight",
        type=float,
        default=1.0,
        help="Coefficient on the auxiliary depth-prediction loss (hyperbolic "
        "only). The aux head sees only ||h|| and predicts node-layer; "
        "forces radial differentiation. 0 disables.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--limit-batches", type=int, default=0, help="0 = no limit")
    p.add_argument("--out", type=str, default="runs/default")
    p.add_argument("--cuda", action="store_true")
    p.add_argument(
        "--train-frac",
        type=float,
        default=1.0,
        help="Fraction of the train split to use. Nested subsets via cfg.seed — "
        "25%% ⊂ 50%% ⊂ 75%% ⊂ 100%%. Used for the learning-curve diagnostic.",
    )
    p.add_argument(
        "--task-loss-weights",
        type=str,
        default="",
        help="Comma-separated per-task multipliers 'w0,w1,w2,w3,w4'. Missing "
        "entries default to 1.0. Use e.g. '2,1,1,0.5,1' to upweight Task 0 "
        "(provenance / 1/d depth) relative to Task 3 (multi-hop binary) and "
        "test whether radial-axis competition is the primary driver of "
        "Task 0 volatility.",
    )
    p.add_argument(
        "--hierarchy-subspace-dim",
        type=int,
        default=0,
        help="If > 0 (hyperbolic only), reserve the first k tangent-at-origin "
        "coordinates for Task 0 and route Tasks 1-4 through the remaining "
        "hidden_dim-k coordinates via a separate scoring head. Gives Task 0's "
        "graded 1/d depth signal a protected radial axis that Task 3's binary "
        "gradient cannot flatten. 0 = disabled (single shared head).",
    )
    p.add_argument(
        "--grad-diag",
        action="store_true",
        help="Log per-bucket gradient norms (manifold / edge_attn / head) at "
        "every --log-every step. Off by default.",
    )
    p.add_argument(
        "--log-depth-diagnostics",
        action="store_true",
        help="Per-round ||h|| and per-task gradient norms across message-passing "
        "rounds. Requires model to return per_round_embeddings; no-op otherwise. "
        "AttnRes Phase-1 diagnostic for depth-wise signal dilution.",
    )
    p.add_argument(
        "--concat-depth",
        action="store_true",
        help="HMT-style diagnostic: score on [logmap0(h_1) || ... || logmap0(h_L)] "
        "instead of just the final round. Widens scoring-head inputs by num_layers. "
        "If multi-hop tasks lift, final-round-only heads were collapsing depth "
        "signal and full AttnRes is justified.",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color coding in terminal output.",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
