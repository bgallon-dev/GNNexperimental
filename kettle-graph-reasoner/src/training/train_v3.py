r"""Two-stage trainer for KGR v3.

Stage A — Contrastive pretraining (graph encoder only).
    Ignores ``sample.query`` / ``sample.labels``. For each graph,
    ``PositiveSampler`` draws anchor/positive pairs (edge + same-label-
    different-features). Loss: InfoNCE on Poincaré-ball distances +
    radial-reg decay. Optimizer: RiemannianAdam when available.

Stage B — Query alignment (``QueryToBall`` head only).
    Graph encoder is frozen (``requires_grad=False``). Loss: pairwise
    ranking on the per-task labels — NOT MSE. See
    ``src/modelsv3/ranking.py`` for the rationale (pointwise regression
    reintroduces the mean-collapse failure mode stage A was designed to
    escape).

Eval at the end: for each val sample, compute node embeddings (frozen
encoder) + query point → per-node score via negative distance →
``MetricAccumulator``. Writes ``summary.json`` in the same shape as
``train.py`` so existing harnesses and downstream scripts keep working.

Optional: ``--include-intrinsic`` dumps silhouette / nn-edge-precision /
nn-label-purity on a single val graph for a quick intrinsic sanity
check without running the full ``intrinsic_eval.py`` script separately.

Flags mirror ``train.py`` where they overlap (``--radial-reg-weight``,
``--radial-reg-weight-end``, ``--tangent-scale``, ``--lr``, ``--seed``,
``--cuda``, ``--log-every``, ``--out``, ``--corpus``, ``--task``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

# Optional: registers the "privateuseone" backend for AMD/Intel GPUs on
# Windows. No-op when torch-directml is not installed.
try:  # pragma: no cover
    import torch_directml  # noqa: F401
except ImportError:
    pass

# Make `src.` importable when invoked from the project root or via -m.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset, Sample
from src.modelsv3.contrastive import PositiveSampler, poincare_infonce
from src.modelsv3.distance_scoring import score_from_embeddings
from src.modelsv3.euclidean_v3 import EuclideanReasonerV3
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3
from src.modelsv3.intrinsic_eval import (
    nn_edge_precision_at_k,
    nn_label_purity_at_k,
    silhouette_score,
)
from src.modelsv3.query_encoder import QueryToBall
from src.modelsv3.stage_b_bilinear import (
    BilinearStageBHead,
    bilinear_listwise_loss,
    bilinear_pairwise_loss,
)
from src.modelsv3.ranking import (
    listwise_ranking_loss,
    pairwise_ranking_loss,
    sampled_infonce_ranking_loss,
)
from src.modelsv3.topology_reg import edge_preserve_loss, radius_stability_loss
from src.modelsv2.layers import poincare_ops as P
from src.training.metrics import MetricAccumulator


NODE_TYPE_SLICE = slice(0, 12)

# Absolute columns of the node temporal SCOPE (start, end) in the 32-D node
# feature vector. Source of truth: src/data/feature_encoder.encode_nodes
# (offset 12+4+5 = 21; col 23 = duration is excluded — an exact affine fn of
# 21,22). Defined here rather than imported because feature_encoder.py uses
# bare sibling imports that do not resolve as a package from this module's
# sys.path. Cross-check: scripts/probe_*.py hardcode TS_COL,TE_COL = 21,22.
NODE_TEMPORAL_COLS = (21, 22)


@dataclass
class Config:
    corpus: str
    task: int
    model: str
    out: str
    hidden_dim: int = 32
    num_layers: int = 3
    type_dim: int = 8
    contrastive_epochs: int = 5
    query_epochs: int = 3
    lr: float = 1e-3
    seed: int = 0
    cuda: bool = False
    device: str = ""  # explicit override (e.g. "privateuseone:0" for DirectML)
    stage_a_n_neg_sample: int = 0  # 0 = full-N softmax; > 0 enables sampled InfoNCE
    # Cross-graph negative queue (MoCo-style): keep ~512 detached rows per
    # recent graph, up to this many total, appended to every softmax.
    # Requires stage_a_n_neg_sample > 0. 0 = off (bit-exact legacy).
    stage_a_xgraph_queue: int = 0
    resume_from: str = ""           # path to a checkpoint produced by this script
    checkpoint_every_epochs: int = 1  # cadence of mid-training checkpoint writes
    throughput_log_every: int = 100   # 0 to disable; otherwise log steps/sec every N
    log_every: int = 50
    tangent_scale: float = 0.1
    radial_reg_weight: float = 0.01
    radial_reg_weight_end: Optional[float] = 0.001
    temperature: float = 1.0
    anchors_per_step: int = 64
    positive_mix: float = 0.5
    use_tangent_approx: bool = False
    neighbor_exclude_k: int = 1
    margin: float = 0.5
    stage_b_loss: str = "pairwise"
    # v3.1 Phase 3 — un-hardcoded stage-B knobs + sampled InfoNCE
    stage_b_n_pairs: int = 16
    stage_b_pos_threshold: float = 0.75
    stage_b_neg_threshold: float = 0.25
    stage_b_negatives: int = 128
    stage_b_n_positives: int = 8
    stage_b_temperature: Optional[float] = None
    train_frac: float = 1.0
    train_graphs_frac: float = 1.0
    graphs_split_seed: int = 0
    uniformity_reg_weight: float = 0.0
    uniformity_t: float = 2.0
    temporal_aux_weight: float = 0.0
    stage_b_head: str = "qtb"
    include_intrinsic: bool = True
    curvature: float = 1.0
    # v3.1 Phase 2 — query-head sweep / frozen-encoder path
    query_head_arch: str = "qh0"
    query_head_norm: str = "layernorm"
    lr_query: Optional[float] = None
    skip_stage_a: bool = False
    load_encoder: Optional[str] = None
    assert_encoder_sha: Optional[str] = None
    # E5 (Docs/ARCH_EFFICIENCY_PLAN.md): whether EdgeTypedAttention layers
    # allocate their internal type_emb table. The v3 forward always passes
    # the SchemaEncoder override, so the table is dead weight (960 params
    # at l4) — but every pre-2026-07-10 checkpoint contains it and must
    # keep loading strict. Dataclass default True so Config(**old_summary_
    # config) reconstructs legacy checkpoints; the CLI defaults to False
    # (new runs are table-free) via --legacy-attn-type-table.
    attn_type_table: bool = True
    # v3.1 Phase 4 — opt-in Stage C (conservative top-layer fine-tune)
    freeze_mode: str = "full"
    stage_c_epochs: int = 0
    lr_encoder: float = 3e-5
    edge_preserve_weight: float = 0.1
    radius_stability_weight: float = 0.1
    baseline_encoder: Optional[str] = None
    # Patch 2 — bf16 autocast around encoder forward (CUDA only).
    no_autocast: bool = False
    # Patch 3 — early-stop Stage-A on gap plateau.
    early_stop: bool = False


def _build_encoder(cfg: Config, dataset: CorpusDataset):
    # attn_type_table=False -> EdgeTypedAttention gets num_edge_types=None
    # and allocates no internal table (the schema override is always
    # supplied in forward). See Config.attn_type_table.
    num_edge_types = dataset.num_edge_types_max if cfg.attn_type_table else None
    if cfg.model == "hyperbolic":
        return KettleGraphReasonerV3(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            type_dim=cfg.type_dim,
            c=cfg.curvature,
            num_edge_types_max=num_edge_types,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
            tangent_scale_init=cfg.tangent_scale,
        )
    if cfg.model == "euclidean":
        return EuclideanReasonerV3(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            type_dim=cfg.type_dim,
            num_edge_types_max=num_edge_types,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
        )
    raise ValueError(f"unknown --model {cfg.model!r}; use hyperbolic or euclidean")


def _encode(model, sample: Sample) -> Tensor:
    out = model(
        sample.x,
        sample.edge_index,
        sample.edge_type,
        sample.edge_descriptor,
        node_descriptor=sample.node_descriptor,
    )
    return out.node_embeddings


def _make_optimizer(model, lr: float, hyperbolic: bool):
    if hyperbolic:
        try:
            from geoopt.optim import RiemannianAdam

            return RiemannianAdam(model.parameters(), lr=lr), "RiemannianAdam"
        except ImportError:
            pass
    return torch.optim.Adam(model.parameters(), lr=lr), "Adam"


def _uniformity_loss(
    node_emb: Tensor, c, euclidean: bool, t: float = 2.0
) -> Tensor:
    """Wang-Isola uniformity loss over all within-graph node pairs.

    For Euclidean models: computed directly on embeddings.
    For hyperbolic models: computed on logmap0 (tangent-at-origin)
    coordinates. This is the standard way to apply Euclidean-style
    regularizers to Poincaré embeddings and keeps both geometries on
    symmetric footing for the sweep.

    Returns ``log E[exp(-t * d(x_i, x_j)^2)]`` where d is L2.
    Minimized when embeddings are uniformly distributed. Diagonal
    (self-pairs) is masked out.

    Gradient flows through node_emb so the encoder learns to spread
    embeddings apart.
    """
    if node_emb.size(0) < 2:
        # degenerate graph, no pairs — return zero tensor in correct graph
        return torch.zeros((), device=node_emb.device, dtype=node_emb.dtype)

    if euclidean:
        coords = node_emb
    else:
        # Batched logmap0 — P.logmap0 expects (..., d) tensor and curvature.
        coords = P.logmap0(node_emb, c)

    # Pairwise squared L2. (N, d) -> (N, N)
    sq_dists = torch.cdist(coords, coords, p=2.0) ** 2
    N = coords.size(0)

    # Mask the diagonal (i == j pairs contribute exp(0) = 1 which would
    # dominate the log-mean and leave no gradient to shape the off-diag).
    mask = ~torch.eye(N, dtype=torch.bool, device=coords.device)
    off_diag = sq_dists[mask]

    # log E[exp(-t * d^2)] computed in a numerically stable way via logsumexp.
    # log_mean = logsumexp(-t*d^2) - log(n_pairs)
    log_mean = torch.logsumexp(-t * off_diag, dim=0) - np.log(off_diag.numel())
    return log_mean


class TemporalReconstructionHead(torch.nn.Module):
    """Stage-A temporal-retention auxiliary head — OPT-IN, and discarded
    after Stage-A (it is never a submodule of the encoder, never in
    ``encoder.state_dict()``, so the frozen-encoder probes' strict
    ``load_state_dict`` is unaffected).

    ``logmap0(h) -> (ts, te)``. Deliberately tiny: the goal is retention
    gradient pressure on the encoder, not a high-capacity temporal
    regressor (same philosophy as the v1 ``DepthHead`` in
    ``src/training/train.py``). Euclidean models do not produce ball
    points, so logmap0 is skipped there (mirrors the ``_uniformity_loss``
    ``euclidean=(not hyperbolic)`` branch).
    """

    def __init__(self, hidden_dim: int, n_targets: int = 2,
                 hidden: int = 64) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, n_targets),
        )

    def forward(self, node_emb: Tensor, c, euclidean: bool = False) -> Tensor:
        coords = node_emb if euclidean else P.logmap0(node_emb, c)
        return self.net(coords)


def _stage_a(cfg: Config, model, dataset: CorpusDataset, device) -> list[dict]:
    from collections import deque as _deque
    if cfg.stage_a_xgraph_queue > 0 and cfg.stage_a_n_neg_sample <= 0:
        raise ValueError("--stage-a-xgraph-queue requires --stage-a-n-neg-sample > 0")
    _xqueue = _deque(maxlen=max(cfg.stage_a_xgraph_queue // 512, 1))
    """Contrastive pretraining. Iterates over all samples in the dataset
    (ignoring their labels / queries); each sample yields one graph, one
    positive-sampler step."""
    hyperbolic = cfg.model == "hyperbolic"
    opt, opt_name = _make_optimizer(model, cfg.lr, hyperbolic=hyperbolic)
    print(f"[stage-A] optimizer: {opt_name}  |  lr={cfg.lr}")

    # Temporal-retention auxiliary (opt-in; default-off => byte-identical to
    # the locked v3.1 baseline). When off: no module, no params, no torch
    # global-RNG consumption, optimizer object unchanged. The RNG
    # save/restore makes head weight-init provably consume zero of the
    # global stream so the aux arm's contrastive path stays comparable to
    # the control's (poincare_infonce uses pre-sampled indices from a numpy
    # RNG and is deterministic — verified).
    temporal_head = None
    if cfg.temporal_aux_weight > 0.0:
        _rng_state = torch.get_rng_state()
        temporal_head = TemporalReconstructionHead(
            cfg.hidden_dim, n_targets=len(NODE_TEMPORAL_COLS)).to(device)
        torch.set_rng_state(_rng_state)
        temporal_head.train()
        opt.add_param_group(
            {"params": list(temporal_head.parameters()), "lr": cfg.lr})
        print(f"[stage-A][aux] temporal-retention head ON  "
              f"w={cfg.temporal_aux_weight}  cols={NODE_TEMPORAL_COLS}  "
              f"(logmap0 in, MSE; discarded after stage-A)")

    rng = np.random.default_rng(cfg.seed)
    # Build a sampler per *unique graph* (not per sample — samples within the
    # same graph share structure). Indexing by file path is robust against
    # the task-to-graph many-to-one mapping.
    sampler_cache: dict[int, PositiveSampler] = {}

    def get_sampler(graph_idx: int, x_t: Tensor, ei_t: Tensor) -> PositiveSampler:
        if graph_idx not in sampler_cache:
            sampler_cache[graph_idx] = PositiveSampler(
                x=x_t.detach().cpu().numpy().astype(np.float32),
                edge_index=ei_t.detach().cpu().numpy().astype(np.int64),
                neighbor_exclude_k=cfg.neighbor_exclude_k,
                edge_fraction=cfg.positive_mix,
                low_cos_threshold=0.4,
                rng=np.random.default_rng(cfg.seed + 17 + graph_idx),
            )
        return sampler_cache[graph_idx]

    steps_per_epoch = len(dataset)
    total_steps = steps_per_epoch * cfg.contrastive_epochs
    reg_start = cfg.radial_reg_weight
    reg_end = reg_start if cfg.radial_reg_weight_end is None else cfg.radial_reg_weight_end

    # Resume from a previous Stage-A checkpoint (epoch-boundary granularity).
    # Restore model + optimizer + numpy/torch RNG so the per-epoch
    # permutation order continues identically; skip already-completed
    # epochs. Per-graph sampler RNGs are seeded deterministically from
    # (cfg.seed, graph_idx) so they reproduce on cache-miss.
    history: list[dict] = []
    start_epoch = 0
    if cfg.resume_from:
        ckpt = torch.load(cfg.resume_from, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
            rng = np.random.default_rng(cfg.seed)  # re-seed local rng
            for _ in range(ckpt["epoch"] + 1):
                rng.permutation(steps_per_epoch)   # advance to match
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        print(f"[stage-A][resume] from {cfg.resume_from} "
              f"@ epoch={ckpt['epoch']}  step={ckpt['step']}  "
              f"=> start_epoch={start_epoch}")

    step = start_epoch * steps_per_epoch
    out_dir = Path(cfg.out)
    t_step_window = time.perf_counter()
    # Patch 2 — bf16 autocast around encoder forward on CUDA only.
    # Manifold ops (poincare_infonce, radial reg, expmap/logmap) run in fp32
    # since bf16 underflows on the boundary-attractor numerics. We cast the
    # encoder output back to fp32 before the InfoNCE block.
    _autocast_on = (device.type == "cuda") and not cfg.no_autocast
    if _autocast_on:
        print(f"[stage-A] bf16 autocast on for encoder forward "
              f"(manifold ops kept in fp32)")
    # Patch 3 — early-stop tracking.
    recent_gaps: list[float] = []
    epoch_gap_sum = 0.0
    epoch_gap_n = 0
    for epoch in range(start_epoch, cfg.contrastive_epochs):
        order = rng.permutation(steps_per_epoch)
        epoch_gap_sum = 0.0
        epoch_gap_n = 0
        for batch_idx in order:
            sample: Sample = dataset[int(batch_idx)]
            sample = _sample_to_device(sample, device)
            graph_idx = int(dataset.index[int(batch_idx)][0])
            sampler = get_sampler(graph_idx, sample.x, sample.edge_index)
            batch = sampler.sample(cfg.anchors_per_step)

            if _autocast_on:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    node_emb = _encode(model, sample)
                node_emb = node_emb.float()
            else:
                node_emb = _encode(model, sample)
            _xq_neg = None
            if cfg.stage_a_xgraph_queue > 0 and _xqueue:
                _xq_neg = torch.cat(list(_xqueue), dim=0).to(node_emb.device)
            loss_info, diag = poincare_infonce(
                node_emb=node_emb,
                anchor_idx=torch.from_numpy(batch.anchor_idx).to(device),
                positive_idx=torch.from_numpy(batch.positive_idx).to(device),
                valid_mask=torch.from_numpy(batch.valid_mask).to(device),
                c=getattr(model, "c", torch.tensor(cfg.curvature)),
                temperature=cfg.temperature,
                use_tangent_approx=cfg.use_tangent_approx,
                n_neg_sample=cfg.stage_a_n_neg_sample,
                extra_neg_emb=_xq_neg,
            )
            if cfg.stage_a_xgraph_queue > 0:
                with torch.no_grad():
                    n_keep = min(512, node_emb.shape[0])
                    sel = torch.randperm(node_emb.shape[0])[:n_keep]
                    _xqueue.append(node_emb[sel].detach().cpu())
            frac = step / max(total_steps - 1, 1)
            reg_w = reg_start + frac * (reg_end - reg_start)
            if hyperbolic:
                radial = (node_emb.norm(dim=-1, p=2) ** 2).mean()
                total = loss_info + reg_w * radial
            else:
                total = loss_info

            # Uniformity regularization (intervention B). Pushes negative
            # pairs apart to prevent origin-attractor collapse (Euclidean)
            # and boundary concentration (hyperbolic). Same form for both
            # geometries via logmap0 for hyperbolic.
            if cfg.uniformity_reg_weight > 0.0:
                unif = _uniformity_loss(
                    node_emb,
                    c=getattr(model, "c", torch.tensor(cfg.curvature)),
                    euclidean=(not hyperbolic),
                    t=cfg.uniformity_t,
                )
                total = total + cfg.uniformity_reg_weight * unif
                unif_val = float(unif.detach())
            else:
                unif_val = 0.0

            # Temporal-retention auxiliary loss (opt-in). MSE of a tiny
            # logmap0->(ts,te) head reconstructing the node temporal-scope
            # columns; pressures the encoder to retain ranking-grade
            # temporal fidelity. Skipped entirely when the head is off.
            if temporal_head is not None:
                t_target = sample.x[:, list(NODE_TEMPORAL_COLS)]
                t_pred = temporal_head(
                    node_emb,
                    getattr(model, "c", torch.tensor(cfg.curvature)),
                    euclidean=(not hyperbolic),
                )
                aux = torch.nn.functional.mse_loss(t_pred, t_target)
                total = total + cfg.temporal_aux_weight * aux
                aux_val = float(aux.detach())
            else:
                aux_val = 0.0

            opt.zero_grad()
            total.backward()
            # Identical to clip_grad_norm_(model.parameters(), 1.0) when the
            # aux head is off (order-invariant L2 over the same param set).
            clip_params = list(model.parameters())
            if temporal_head is not None:
                clip_params += list(temporal_head.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            opt.step()

            if not torch.isfinite(total):
                raise RuntimeError(
                    f"[stage-A] non-finite loss at epoch {epoch} step {step}"
                )
            if step % cfg.log_every == 0 or step == total_steps - 1:
                print(
                    f"[stage-A] epoch {epoch}  step {step:5d}  "
                    f"loss={diag['loss']:.4f}  "
                    f"pos={diag['mean_pos_sim']:+.3f}  "
                    f"neg={diag['mean_neg_sim']:+.3f}  "
                    f"gap={diag['mean_pos_sim'] - diag['mean_neg_sim']:+.3f}  "
                    f"|h|mean={diag['mean_h_norm']:.3f}  "
                    f"|h|max={diag['max_h_norm']:.3f}  "
                    f"eff_negs={diag['eff_negs_per_anchor']:.0f}  "
                    f"reg_w={reg_w:.4f}  "
                    f"unif={unif_val:+.4f}  "
                    f"aux={aux_val:+.4f}"
                )
            history.append(
                {"step": step, "epoch": epoch, **diag,
                 "reg_w": reg_w, "uniformity": unif_val,
                 "temporal_aux": aux_val}
            )
            # Patch 3 — accumulate per-step gap for end-of-epoch plateau check.
            epoch_gap_sum += float(diag["mean_pos_sim"] - diag["mean_neg_sim"])
            epoch_gap_n += 1
            step += 1

            # Throughput probe: steps/sec over the last window + GPU util
            # when available. Cheap; off by default at log_every=0.
            if cfg.throughput_log_every and step % cfg.throughput_log_every == 0:
                dt = time.perf_counter() - t_step_window
                sps = cfg.throughput_log_every / max(dt, 1e-6)
                util_str = ""
                if torch.cuda.is_available() and device.type == "cuda":
                    try:
                        u = torch.cuda.utilization(device)
                        util_str = f"  cuda_util={u}%"
                    except Exception:  # noqa
                        pass
                print(f"[stage-A][thru] step {step}  "
                      f"steps/sec={sps:.2f}{util_str}", flush=True)
                t_step_window = time.perf_counter()

        # End-of-epoch checkpoint (atomic). Keeps the run resumable from
        # the last completed epoch boundary; intra-epoch kill loses at
        # most one epoch of work. Per-graph sampler RNGs reseed
        # deterministically on cache-miss after resume.
        if (cfg.checkpoint_every_epochs > 0
                and (epoch + 1) % cfg.checkpoint_every_epochs == 0):
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / "stage_a_state.pt.tmp"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "step": step, "epoch": epoch,
                "history": history,
                "cfg_signature": {
                    "hidden_dim": cfg.hidden_dim,
                    "num_layers": cfg.num_layers,
                    "model": cfg.model,
                    "seed": cfg.seed,
                    "stage_a_n_neg_sample": cfg.stage_a_n_neg_sample,
                },
            }, tmp)
            tmp.replace(out_dir / "stage_a_state.pt")
            print(f"[stage-A][ckpt] wrote {out_dir / 'stage_a_state.pt'} "
                  f"@ epoch {epoch}", flush=True)

        # Patch 3 — early-stop on gap plateau. Compare last-3 epoch mean gap
        # against the prior 3; stop if relative improvement < 1%. min_epochs=6.
        if cfg.early_stop and epoch_gap_n > 0:
            recent_gaps.append(epoch_gap_sum / epoch_gap_n)
            if len(recent_gaps) >= 6:
                early = sum(recent_gaps[-3:]) / 3.0
                old = sum(recent_gaps[-6:-3]) / 3.0
                rel = (early - old) / max(abs(old), 1e-6)
                if rel < 0.01:
                    print(f"[stage-A] early-stop at epoch {epoch} "
                          f"(gap plateau: last-3 mean={early:.4f}, "
                          f"prior-3 mean={old:.4f}, rel={rel:.3%})",
                          flush=True)
                    break

    return history


def _stage_b(
    cfg: Config,
    model,
    query_encoder: QueryToBall,
    dataset: CorpusDataset,
    device,
) -> list[dict]:
    """Query alignment with a frozen graph encoder."""
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    lr_q = cfg.lr if cfg.lr_query is None else cfg.lr_query
    opt = torch.optim.Adam(query_encoder.parameters(), lr=lr_q)
    print(f"[stage-B] query-head lr={lr_q}  arch={cfg.query_head_arch}")
    euclidean = cfg.model == "euclidean"

    rng = np.random.default_rng(cfg.seed + 101)
    n = len(dataset)
    if cfg.train_frac < 1.0:
        k = max(1, int(round(cfg.train_frac * n)))
        subset = rng.choice(n, size=k, replace=False)
    else:
        subset = np.arange(n)
    print(f"[stage-B] {len(subset)}/{n} samples used (train_frac={cfg.train_frac})")

    total_steps = len(subset) * cfg.query_epochs
    step = 0
    history: list[dict] = []
    for epoch in range(cfg.query_epochs):
        order = subset[rng.permutation(len(subset))]
        for batch_idx in order:
            sample = _sample_to_device(dataset[int(batch_idx)], device)
            with torch.no_grad():
                node_emb = _encode(model, sample)
            c_val = getattr(model, "c", torch.tensor(cfg.curvature))
            # Stage-B temperature is decoupled from Stage-A's --temperature.
            # When unset it falls back to --temperature so listwise is
            # bit-identical to pre-v3.1 behaviour.
            sb_temp = (
                cfg.temperature
                if cfg.stage_b_temperature is None
                else cfg.stage_b_temperature
            )
            if cfg.stage_b_head == "bilinear":
                # Opt-in bilinear path: head emits scores directly; faithful
                # score-based losses replicated from the validated probes
                # (ranking.py untouched). infonce never validated here.
                scores = query_encoder(sample.query, node_emb)
                if cfg.stage_b_loss == "pairwise":
                    loss, diag = bilinear_pairwise_loss(
                        scores, sample.labels,
                        margin=cfg.margin,
                        n_pairs=cfg.stage_b_n_pairs,
                        pos_threshold=cfg.stage_b_pos_threshold,
                        neg_threshold=cfg.stage_b_neg_threshold,
                    )
                elif cfg.stage_b_loss == "listwise":
                    loss, diag = bilinear_listwise_loss(
                        scores, sample.labels, temperature=sb_temp,
                    )
                else:
                    raise ValueError(
                        f"stage_b_head=bilinear supports stage_b_loss in "
                        f"{{pairwise, listwise}}; got {cfg.stage_b_loss!r} "
                        f"(infonce never validated for the bilinear head)"
                    )
            else:
                q_point = query_encoder(sample.query)
                if cfg.stage_b_loss == "pairwise":
                    loss, diag = pairwise_ranking_loss(
                        query_point=q_point,
                        node_embeddings=node_emb,
                        labels=sample.labels,
                        c=c_val,
                        margin=cfg.margin,
                        n_pairs=cfg.stage_b_n_pairs,
                        pos_threshold=cfg.stage_b_pos_threshold,
                        neg_threshold=cfg.stage_b_neg_threshold,
                        euclidean=euclidean,
                    )
                elif cfg.stage_b_loss == "listwise":
                    loss, diag = listwise_ranking_loss(
                        query_point=q_point,
                        node_embeddings=node_emb,
                        labels=sample.labels,
                        c=c_val,
                        temperature=sb_temp,
                        euclidean=euclidean,
                    )
                elif cfg.stage_b_loss == "infonce":
                    loss, diag = sampled_infonce_ranking_loss(
                        query_point=q_point,
                        node_embeddings=node_emb,
                        labels=sample.labels,
                        c=c_val,
                        n_negatives=cfg.stage_b_negatives,
                        temperature=sb_temp,
                        n_positives=cfg.stage_b_n_positives,
                        pos_threshold=cfg.stage_b_pos_threshold,
                        neg_threshold=cfg.stage_b_neg_threshold,
                        euclidean=euclidean,
                    )
                else:
                    raise ValueError(
                        f"unknown stage_b_loss: {cfg.stage_b_loss!r}")

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), 1.0)
            opt.step()

            if not torch.isfinite(loss):
                raise RuntimeError(f"[stage-B] non-finite loss at step {step}")
            if step % cfg.log_every == 0 or step == total_steps - 1:
                extras = " ".join(
                    f"{k}={v:.3f}"
                    for k, v in diag.items()
                    if isinstance(v, (int, float)) and k != "loss"
                )
                print(
                    f"[stage-B] epoch {epoch}  step {step:5d}  "
                    f"loss={diag['loss']:.4f}  {extras}"
                )
            history.append({"step": step, "epoch": epoch, **diag})
            step += 1

    # Unfreeze so eval / downstream code behaves normally.
    for p in model.parameters():
        p.requires_grad = True
    return history


def _unfreeze_last_layer(model) -> list:
    """Freeze the whole encoder EXCEPT the final attention + message-
    passing layer AND ``depth_attention``.

    Critical (v3.1 plan risk #2): the final ``node_embeddings`` are
    produced by ``depth_attention`` (hyperbolic_gnnV3.py:224-231), NOT
    by ``mp_layers[-1]`` directly. Unfreezing only ``mp_layers[-1]``
    would leave the true output projection frozen. Returns the list of
    now-trainable encoder parameters for the optimizer group.
    """
    for p in model.parameters():
        p.requires_grad = False
    unfrozen: list = []
    targets = []
    if hasattr(model, "attn_layers") and len(model.attn_layers) > 0:
        targets.append(model.attn_layers[-1])
    if hasattr(model, "mp_layers") and len(model.mp_layers) > 0:
        targets.append(model.mp_layers[-1])
    da = getattr(model, "depth_attention", None)
    if da is not None:
        targets.append(da)
    for mod in targets:
        for p in mod.parameters():
            p.requires_grad = True
            unfrozen.append(p)
    return unfrozen


def _stage_c(
    cfg: Config,
    model,
    query_encoder: QueryToBall,
    baseline_encoder,
    dataset: CorpusDataset,
    device,
) -> list[dict]:
    """Conservative joint fine-tune tail (OPT-IN, runs AFTER a converged
    stage B — never instead of it). Unfreezes only the top encoder layer
    + ``depth_attention`` and the query head; clamps drift with the
    topology-preservation penalties measured against ``baseline_encoder``
    (the frozen structural ground truth)."""
    enc_params = _unfreeze_last_layer(model)
    model.train()
    for p in query_encoder.parameters():
        p.requires_grad = True
    baseline_encoder.eval()
    for p in baseline_encoder.parameters():
        p.requires_grad = False

    lr_q = cfg.lr if cfg.lr_query is None else cfg.lr_query
    opt = torch.optim.Adam(
        [
            {"params": query_encoder.parameters(), "lr": lr_q},
            {"params": enc_params, "lr": cfg.lr_encoder},
        ]
    )
    n_enc = sum(p.numel() for p in enc_params)
    print(f"[stage-C] unfrozen encoder params: {n_enc}  "
          f"lr_encoder={cfg.lr_encoder}  lr_query={lr_q}  "
          f"lambda_edge={cfg.edge_preserve_weight} "
          f"lambda_radius={cfg.radius_stability_weight}")

    euclidean = cfg.model == "euclidean"
    rng = np.random.default_rng(cfg.seed + 202)
    n = len(dataset)
    base_cache: dict[int, Tensor] = {}
    history: list[dict] = []
    step = 0
    total_steps = n * cfg.stage_c_epochs
    for epoch in range(cfg.stage_c_epochs):
        order = rng.permutation(n)
        for batch_idx in order:
            gi = int(dataset.index[int(batch_idx)][0])
            sample = _sample_to_device(dataset[int(batch_idx)], device)
            if gi not in base_cache:
                with torch.no_grad():
                    base_cache[gi] = _encode(baseline_encoder, sample).detach()
            emb_base = base_cache[gi]

            node_emb = _encode(model, sample)
            q_point = query_encoder(sample.query)
            c_val = getattr(model, "c", torch.tensor(cfg.curvature))
            sb_temp = (
                cfg.temperature
                if cfg.stage_b_temperature is None
                else cfg.stage_b_temperature
            )
            if cfg.stage_b_loss == "pairwise":
                rank_loss, diag = pairwise_ranking_loss(
                    q_point, node_emb, sample.labels, c=c_val,
                    margin=cfg.margin, n_pairs=cfg.stage_b_n_pairs,
                    pos_threshold=cfg.stage_b_pos_threshold,
                    neg_threshold=cfg.stage_b_neg_threshold,
                    euclidean=euclidean,
                )
            elif cfg.stage_b_loss == "listwise":
                rank_loss, diag = listwise_ranking_loss(
                    q_point, node_emb, sample.labels, c=c_val,
                    temperature=sb_temp, euclidean=euclidean,
                )
            elif cfg.stage_b_loss == "infonce":
                rank_loss, diag = sampled_infonce_ranking_loss(
                    q_point, node_emb, sample.labels, c=c_val,
                    n_negatives=cfg.stage_b_negatives, temperature=sb_temp,
                    n_positives=cfg.stage_b_n_positives,
                    pos_threshold=cfg.stage_b_pos_threshold,
                    neg_threshold=cfg.stage_b_neg_threshold,
                    euclidean=euclidean,
                )
            else:
                raise ValueError(f"unknown stage_b_loss {cfg.stage_b_loss!r}")

            edge_pres = edge_preserve_loss(
                node_emb, emb_base, sample.edge_index, c=c_val,
                euclidean=euclidean,
            )
            rad_stab = radius_stability_loss(
                node_emb, emb_base, c=c_val, euclidean=euclidean,
            )
            total = (
                rank_loss
                + cfg.edge_preserve_weight * edge_pres
                + cfg.radius_stability_weight * rad_stab
            )

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(query_encoder.parameters()) + enc_params, 1.0
            )
            opt.step()

            if not torch.isfinite(total):
                raise RuntimeError(f"[stage-C] non-finite loss at step {step}")
            if step % cfg.log_every == 0 or step == total_steps - 1:
                print(
                    f"[stage-C] epoch {epoch}  step {step:5d}  "
                    f"rank={diag['loss']:.4f}  "
                    f"edge_pres={float(edge_pres):.4f}  "
                    f"rad_stab={float(rad_stab):.4f}  "
                    f"total={float(total):.4f}"
                )
            history.append({
                "step": step, "epoch": epoch,
                "rank_loss": diag["loss"],
                "rank_accuracy": diag.get("rank_accuracy", float("nan")),
                "edge_preserve": float(edge_pres.detach()),
                "radius_stability": float(rad_stab.detach()),
                "total": float(total.detach()),
            })
            step += 1

    for p in model.parameters():
        p.requires_grad = True
    return history


def _eval(cfg: Config, model, query_encoder: QueryToBall, dataset: CorpusDataset, device) -> dict:
    acc = MetricAccumulator()
    euclidean = cfg.model == "euclidean"
    c_val = getattr(model, "c", torch.tensor(cfg.curvature))
    model.eval()
    query_encoder.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = _sample_to_device(dataset[i], device)
            node_emb = _encode(model, sample)
            if cfg.stage_b_head == "bilinear":
                scores = query_encoder(sample.query, node_emb)
            else:
                q_point = query_encoder(sample.query)
                scores = score_from_embeddings(
                    node_embeddings=node_emb,
                    query_point=q_point,
                    c=c_val,
                    euclidean=euclidean,
                )
            acc.add(scores.detach().cpu(), sample.labels.detach().cpu(), sample.task_type)
    return acc.summary()


def _intrinsic_probe(cfg: Config, model, dataset: CorpusDataset, device) -> dict:
    """One-graph intrinsic probe on a held-out sample. Fast sanity check."""
    if len(dataset) == 0:
        return {}
    sample = _sample_to_device(dataset[0], device)
    with torch.no_grad():
        node_emb = _encode(model, sample)
    euclidean = cfg.model == "euclidean"
    c_val = getattr(model, "c", torch.tensor(cfg.curvature))
    type_block = sample.x[:, NODE_TYPE_SLICE]
    sums = type_block.sum(dim=1)
    type_labels = torch.where(
        sums > 0, type_block.argmax(dim=1), torch.full_like(sums, -1, dtype=torch.long)
    )
    return {
        "silhouette_node_type": silhouette_score(
            node_emb.detach().cpu(), type_labels.detach().cpu(),
            c=c_val, euclidean=euclidean,
        ),
        "nn_edge_precision@5": nn_edge_precision_at_k(
            node_emb.detach().cpu(), sample.edge_index.detach().cpu(),
            k=5, c=c_val, euclidean=euclidean,
        ),
        "nn_label_purity@5": nn_label_purity_at_k(
            node_emb.detach().cpu(), type_labels.detach().cpu(),
            k=5, c=c_val, euclidean=euclidean,
        ),
    }


def _sample_to_device(sample: Sample, device) -> Sample:
    if device.type == "cpu":
        return sample
    return Sample(
        x=sample.x.to(device),
        edge_index=sample.edge_index.to(device),
        edge_type=sample.edge_type.to(device),
        edge_descriptor=sample.edge_descriptor.to(device),
        node_descriptor=sample.node_descriptor.to(device),
        query=sample.query.to(device),
        labels=sample.labels.to(device),
        task_type=sample.task_type,
    )


def _partition_train_graphs(
    dataset: CorpusDataset, frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """Partition the unique graph indices of ``dataset`` into a "used"
    subset (``frac`` of them) and a "held-out" subset. Deterministic
    under ``seed``. Returns sorted lists.

    Operates at graph level, not sample level, so all tasks of a
    held-out graph are held out together (no within-graph leakage).
    """
    seen: set[int] = set()
    unique_graphs: list[int] = []
    for graph_idx, _task_idx in dataset.index:
        gi = int(graph_idx)
        if gi not in seen:
            seen.add(gi)
            unique_graphs.append(gi)
    unique_graphs.sort()

    n = len(unique_graphs)
    if frac >= 1.0 or n == 0:
        return unique_graphs, []
    k = max(1, int(round(frac * n)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    used_indices = sorted(int(perm[i]) for i in range(k))
    held_indices = sorted(int(perm[i]) for i in range(k, n))
    used = sorted(unique_graphs[i] for i in used_indices)
    held = sorted(unique_graphs[i] for i in held_indices)
    return used, held


def _filter_dataset_to_graphs(
    dataset: CorpusDataset, allowed_graph_ids: set[int]
) -> None:
    """In-place filter of ``dataset.index`` to retain only (graph_idx,
    task_idx) tuples whose graph_idx is in ``allowed_graph_ids``.

    This is the minimal surgical change needed to hold out graphs.
    The dataset's ``_get_graph`` cache is unaffected; it lazily
    populates per graph_idx, so held-out graphs simply never get
    loaded when training iterates over the (now-shorter) index.
    """
    dataset.index = [
        (g, t) for (g, t) in dataset.index if int(g) in allowed_graph_ids
    ]


def train(cfg: Config) -> dict:
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device:
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if cfg.cuda and torch.cuda.is_available() else "cpu")
    print(f"[train_v3] device={device}  model={cfg.model}  task={cfg.task}  seed={cfg.seed}")

    train_ds = CorpusDataset(
        corpus_dir=cfg.corpus, split="train", split_seed=0,
        include_tasks={cfg.task},
    )
    val_ds = CorpusDataset(
        corpus_dir=cfg.corpus, split="val", split_seed=0,
        include_tasks={cfg.task},
    )

    # Partition train graphs into "used for training" and "held out".
    # When train_graphs_frac = 1.0 (default), used_graph_ids contains
    # every graph and held_out_graph_ids is empty — no behavioural
    # change from prior runs.
    used_graph_ids, held_out_graph_ids = _partition_train_graphs(
        train_ds, cfg.train_graphs_frac, cfg.graphs_split_seed,
    )
    if cfg.train_graphs_frac < 1.0:
        n_all_samples = len(train_ds)
        _filter_dataset_to_graphs(train_ds, set(used_graph_ids))
        print(
            f"[train_v3] held out {len(held_out_graph_ids)}/"
            f"{len(used_graph_ids) + len(held_out_graph_ids)} train graphs "
            f"(train_graphs_frac={cfg.train_graphs_frac}, "
            f"graphs_split_seed={cfg.graphs_split_seed}); "
            f"samples {len(train_ds)}/{n_all_samples} retained"
        )

    print(f"[train_v3] train={len(train_ds)}  val={len(val_ds)}  query_dim={train_ds.query_dim}")

    encoder = _build_encoder(cfg, train_ds).to(device)
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[train_v3] encoder params: {n_params}")

    if cfg.skip_stage_a:
        # v3.1 Phase 2: reuse one frozen encoder across the query-head
        # sweep. No re-pretrain, no per-arm encoder confound.
        if not cfg.load_encoder:
            raise ValueError("--skip-stage-a requires --load-encoder PATH")
        enc_path = Path(cfg.load_encoder)
        if cfg.assert_encoder_sha:
            from src.modelsv3.lock_baseline import sha256_file
            got = sha256_file(enc_path)
            if got != cfg.assert_encoder_sha:
                raise ValueError(
                    f"--assert-encoder-sha mismatch: {enc_path} has "
                    f"{got[:12]}..., expected {cfg.assert_encoder_sha[:12]}..."
                )
        sd = torch.load(enc_path, map_location=device)
        # E5: table-ness must match the CHECKPOINT, not the CLI default —
        # sniff and rebuild so pre/post-E5 encoders both load strict.
        has_table = any(k.endswith("type_emb.weight") for k in sd)
        if has_table != cfg.attn_type_table:
            print(f"[train_v3][E5] {enc_path} "
                  f"{'has' if has_table else 'lacks'} attn type_emb tables; "
                  f"rebuilding encoder to match the checkpoint.")
            cfg.attn_type_table = has_table
            encoder = _build_encoder(cfg, train_ds).to(device)
            n_params = sum(
                p.numel() for p in encoder.parameters() if p.requires_grad)
        encoder.load_state_dict(sd)
        encoder.eval()
        stage_a_history = []
        print(f"[train_v3] stage A SKIPPED; loaded frozen encoder "
              f"{enc_path} (sha-asserted={bool(cfg.assert_encoder_sha)})")
    else:
        stage_a_history = _stage_a(cfg, encoder, train_ds, device)

    if cfg.stage_b_head == "bilinear":
        # Opt-in bilinear Stage-B head (q^T M node_emb). NOT a QueryToBall:
        # emits scores, not a ball point. Never validated with Stage-C; the
        # default-off path is unaffected.
        if cfg.freeze_mode == "last_layer" and cfg.stage_c_epochs > 0:
            raise NotImplementedError(
                "stage_b_head=bilinear does not support Stage-C "
                "(freeze_mode=last_layer + stage_c_epochs>0)."
            )
        query_encoder = BilinearStageBHead(
            query_dim=train_ds.query_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(device)
    else:
        query_encoder = QueryToBall(
            query_dim=train_ds.query_dim,
            hidden_dim=cfg.hidden_dim,
            c=cfg.curvature,
            euclidean=(cfg.model == "euclidean"),
            arch=cfg.query_head_arch,
            norm=cfg.query_head_norm,
        ).to(device)
    q_params = sum(p.numel() for p in query_encoder.parameters() if p.requires_grad)
    print(f"[train_v3] query encoder params: {q_params}")

    stage_b_history = _stage_b(cfg, encoder, query_encoder, train_ds, device)

    # v3.1 Phase 4 — opt-in conservative Stage C (default: disabled).
    stage_c_history: list[dict] = []
    if cfg.freeze_mode == "last_layer" and cfg.stage_c_epochs > 0:
        import copy
        ref = _build_encoder(cfg, train_ds).to(device)
        if cfg.baseline_encoder:
            ref.load_state_dict(
                torch.load(cfg.baseline_encoder, map_location=device))
            print(f"[stage-C] topology reference = {cfg.baseline_encoder}")
        else:
            ref.load_state_dict(copy.deepcopy(encoder.state_dict()))
            print("[stage-C] topology reference = post-stage-B encoder snapshot")
        ref.eval()
        stage_c_history = _stage_c(
            cfg, encoder, query_encoder, ref, train_ds, device)
    elif cfg.stage_c_epochs > 0 and cfg.freeze_mode != "last_layer":
        print(f"[train_v3] --stage-c-epochs={cfg.stage_c_epochs} ignored: "
              f"requires --freeze-mode last_layer (got {cfg.freeze_mode!r})")

    final_val = _eval(cfg, encoder, query_encoder, val_ds, device)
    final_train = _eval(cfg, encoder, query_encoder, train_ds, device)

    intrinsic = _intrinsic_probe(cfg, encoder, val_ds, device) if cfg.include_intrinsic else {}

    summary = {
        "model": cfg.model,
        "task": cfg.task,
        "seed": cfg.seed,
        "n_params_encoder": n_params,
        "n_params_query": q_params,
        "epochs": {
            "stage_a": 0 if cfg.skip_stage_a else cfg.contrastive_epochs,
            "stage_b": cfg.query_epochs,
            "stage_c": len(stage_c_history) and cfg.stage_c_epochs,
        },
        "final_train": final_train,
        "final_val": final_val,
        "intrinsic_val_graph0": intrinsic,
        "config": asdict(cfg),
        "training_graph_partition": {
            "used_graph_ids": used_graph_ids,
            "held_out_graph_ids": held_out_graph_ids,
            "n_used": len(used_graph_ids),
            "n_held_out": len(held_out_graph_ids),
            "train_graphs_frac": cfg.train_graphs_frac,
            "graphs_split_seed": cfg.graphs_split_seed,
        },
    }
    # Task-by-task gap in the same shape as train.py writes.
    train_by_t = final_train.get("by_task_type", {})
    val_by_t = final_val.get("by_task_type", {})
    summary["gap_by_task"] = {
        str(t): {
            "train_ndcg@10": train_by_t.get(t, {}).get("ndcg@10", float("nan")),
            "val_ndcg@10": m.get("ndcg@10", float("nan")),
            "gap": train_by_t.get(t, {}).get("ndcg@10", float("nan"))
            - m.get("ndcg@10", float("nan")),
        }
        for t, m in val_by_t.items()
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "stage_a_history.json").write_text(json.dumps(stage_a_history, indent=2))
    (out_dir / "stage_b_history.json").write_text(json.dumps(stage_b_history, indent=2))
    if stage_c_history:
        (out_dir / "stage_c_history.json").write_text(
            json.dumps(stage_c_history, indent=2))
    torch.save(encoder.state_dict(), out_dir / "encoder.pt")
    if cfg.stage_b_head == "bilinear":
        # Deliberately DO NOT write query_encoder.pt: this head is not a
        # QueryToBall (-dist) head. Its absence makes every downstream
        # loader (retrieval_ops/neo4j_eval_export, all torch.load
        # query_encoder.pt) fail loudly rather than silently mis-score.
        torch.save(query_encoder.state_dict(), out_dir / "stage_b_head.pt")
    else:
        torch.save(query_encoder.state_dict(),
                   out_dir / "query_encoder.pt")

    print("\n[train_v3] === eval ===")
    print(json.dumps({k: summary[k] for k in ("final_val", "intrinsic_val_graph0")}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--task", type=int, required=True,
                   help="Task type (0..5) to filter stage-B and eval on.")
    p.add_argument("--model", type=str, default="hyperbolic",
                   choices=["hyperbolic", "euclidean"])
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--type-dim", type=int, default=8)
    p.add_argument("--contrastive-epochs", type=int, default=5)
    p.add_argument("--query-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--device", default="",
                   help="explicit device override (e.g. 'privateuseone:0' "
                        "for torch-directml). Beats --cuda when set.")
    p.add_argument("--stage-a-xgraph-queue", type=int, default=0,
                   help="cross-graph negative queue size (0=off; needs "
                        "--stage-a-n-neg-sample>0)")
    p.add_argument("--stage-a-n-neg-sample", type=int, default=0,
                   help="If > 0, sample this many random negatives per "
                        "anchor in Stage-A InfoNCE instead of full-N "
                        "softmax. Required to fit large graphs (N > ~10k) "
                        "at non-trivial hidden_dim without OOM. K=2048 is "
                        "a reasonable default for code-graph scale.")
    p.add_argument("--no-autocast", action="store_true",
                   help="Disable bf16 autocast around the encoder forward "
                        "(Patch 2). Default: autocast on when running on "
                        "CUDA. Manifold ops (poincare_infonce, radial reg) "
                        "always run in fp32.")
    p.add_argument("--early-stop", action="store_true",
                   help="Stop Stage-A early when the InfoNCE gap "
                        "(pos_sim - neg_sim) plateaus over the last 3 "
                        "epochs (<1% relative improvement vs the prior 3). "
                        "min_epochs=6 to avoid stopping on noise.")
    p.add_argument("--resume-from", default="",
                   help="Path to a stage_a_state.pt checkpoint. Restores "
                        "model + optimizer + RNG and resumes at the saved "
                        "epoch boundary. Use with the same Config — only "
                        "skipping over completed epochs is supported.")
    p.add_argument("--checkpoint-every-epochs", type=int, default=1,
                   help="Write stage_a_state.pt every N Stage-A epochs "
                        "(atomically). The last 2 + the encoder.pt at the "
                        "best epoch are kept. Set to 0 to disable mid-run "
                        "checkpoints (final encoder.pt still saves).")
    p.add_argument("--throughput-log-every", type=int, default=100,
                   help="Log steps/sec and cuda.utilization() every N "
                        "Stage-A steps. 0 disables.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--tangent-scale", type=float, default=0.1)
    p.add_argument("--radial-reg-weight", type=float, default=0.01)
    p.add_argument("--radial-reg-weight-end", type=float, default=0.001)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--anchors-per-step", type=int, default=64)
    p.add_argument("--positive-mix", type=float, default=0.5,
                   help="Fraction of positives drawn from the edge signal; "
                        "the rest from same-label-different-features.")
    p.add_argument("--use-tangent-approx", action="store_true")
    p.add_argument("--neighbor-exclude-k", type=int, default=1)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--stage-b-loss", type=str, default="pairwise",
                   choices=["pairwise", "listwise", "infonce"])
    p.add_argument("--stage-b-n-pairs", type=int, default=16,
                   help="Pairs/sample for the pairwise stage-B loss "
                        "(was hardcoded pre-v3.1).")
    p.add_argument("--stage-b-pos-threshold", type=float, default=0.75)
    p.add_argument("--stage-b-neg-threshold", type=float, default=0.25)
    p.add_argument("--stage-b-negatives", type=int, default=128,
                   help="Sampled negative pool size for --stage-b-loss infonce.")
    p.add_argument("--stage-b-n-positives", type=int, default=8,
                   help="Sampled positives/sample for --stage-b-loss infonce.")
    p.add_argument("--stage-b-temperature", type=float, default=None,
                   help="Stage-B softmax temperature (listwise/infonce). "
                        "Decoupled from Stage-A --temperature; defaults to it.")
    p.add_argument("--train-frac", type=float, default=1.0,
                   help="Fraction of train-split stage-B data to use. "
                        "Primary lever for the query-agnosticism sanity check.")
    p.add_argument("--train-graphs-frac", type=float, default=1.0,
                   help="Fraction of UNIQUE TRAIN GRAPHS to use for training. "
                        "When <1.0, the complementary fraction is held out "
                        "entirely (no stage-A or stage-B signal). Records both "
                        "sets in summary.json so Experiment 2.2 (cross-graph "
                        "transfer) can evaluate seen vs unseen-train.")
    p.add_argument("--graphs-split-seed", type=int, default=0,
                   help="Seed for the held-out graph split. Independent of "
                        "--seed so multiple seed runs share the same held-out "
                        "set unless this is explicitly varied.")
    p.add_argument("--uniformity-reg-weight", type=float, default=0.0,
                   help="Weight for Wang-Isola uniformity regularization "
                        "(log E[exp(-t*d^2)]) added to stage-A loss. Pushes "
                        "embedding pairs apart to counter origin-attractor "
                        "collapse (Euclidean) or boundary concentration "
                        "(hyperbolic). Default 0 = disabled. Typical "
                        "sweep: {0.0, 0.01, 0.1}.")
    p.add_argument("--uniformity-t", type=float, default=2.0,
                   help="Temperature parameter for the uniformity loss. "
                        "Default 2.0 matches Wang-Isola. Higher t = sharper "
                        "penalty on close pairs.")
    p.add_argument("--temporal-aux-weight", type=float, default=0.0,
                   help="Opt-in Stage-A temporal-retention auxiliary: MSE of "
                        "a tiny logmap0->(ts,te) head reconstructing node "
                        "temporal-scope cols (21,22) from node embeddings. "
                        "Pressures the encoder to retain ranking-grade "
                        "temporal fidelity. Default 0 = disabled "
                        "(byte-identical to the locked v3.1 baseline).")
    p.add_argument("--stage-b-head", type=str, default="qtb",
                   choices=["qtb", "bilinear"],
                   help="Stage-B scoring head. 'qtb' (default) = QueryToBall "
                        "+ -dist (unchanged, byte-identical). 'bilinear' = "
                        "opt-in q^T M node_emb head (scores, not a ball "
                        "point); writes stage_b_head.pt instead of "
                        "query_encoder.pt; supports stage-b-loss "
                        "{pairwise,listwise} only.")
    p.add_argument("--no-intrinsic", action="store_true",
                   help="Skip the per-run intrinsic probe (silhouette, nn@5).")
    p.add_argument("--curvature", type=float, default=1.0)
    # v3.1 Phase 2 — query-head sweep / frozen-encoder path
    p.add_argument("--query-head-arch", type=str, default="qh0",
                   choices=["qh0", "qh1", "qh2", "qh3", "qh4"],
                   help="QueryToBall architecture. qh0 = pre-v3.1 baseline.")
    p.add_argument("--query-head-norm", type=str, default="layernorm",
                   choices=["layernorm", "rmsnorm"],
                   help="Norm used by the qh2/qh3 variants.")
    p.add_argument("--lr-query", type=float, default=None,
                   help="Stage-B (query head) learning rate. Defaults to --lr.")
    p.add_argument("--skip-stage-a", action="store_true",
                   help="Skip stage-A pretraining; load a frozen encoder "
                        "via --load-encoder. Mandatory for the query-head "
                        "sweep so all arms share one encoder.")
    p.add_argument("--load-encoder", type=str, default=None,
                   help="Path to a frozen encoder.pt (with --skip-stage-a).")
    p.add_argument("--assert-encoder-sha", type=str, default=None,
                   help="Optional sha256 the --load-encoder file must match "
                        "(proves the locked baseline encoder was used).")
    # E5 (Docs/ARCH_EFFICIENCY_PLAN.md): new runs are table-free by
    # default. The flag restores the pre-2026-07-10 construction (dead
    # per-layer type_emb tables) — needed only to resume/extend a pre-E5
    # checkpoint bit-compatibly. --load-encoder auto-detects either way.
    p.add_argument("--legacy-attn-type-table", dest="attn_type_table",
                   action="store_true",
                   help="Allocate the (unused) EdgeTypedAttention internal "
                        "type_emb tables, matching pre-E5 checkpoints. "
                        "Default: off for new runs.")
    # v3.1 Phase 4 — opt-in conservative Stage C
    p.add_argument("--freeze-mode", type=str, default="full",
                   choices=["full", "last_layer"],
                   help="full = encoder fully frozen in stage B (default, "
                        "query-agnostic). last_layer = enable Stage C to "
                        "fine-tune the top encoder layer + depth_attention.")
    p.add_argument("--stage-c-epochs", type=int, default=0,
                   help="Stage-C fine-tune epochs. 0 = disabled (default). "
                        "Only runs with --freeze-mode last_layer.")
    p.add_argument("--lr-encoder", type=float, default=3e-5,
                   help="Stage-C learning rate for the unfrozen top layer.")
    p.add_argument("--edge-preserve-weight", type=float, default=0.1)
    p.add_argument("--radius-stability-weight", type=float, default=0.1)
    p.add_argument("--baseline-encoder", type=str, default=None,
                   help="Encoder.pt used as the Stage-C topology reference. "
                        "Defaults to a post-stage-B snapshot of the encoder.")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"[stage-A] TF32 on (matmul + cudnn), cudnn.benchmark on")
    cfg = Config(
        corpus=a.corpus, task=a.task, model=a.model, out=a.out,
        hidden_dim=a.hidden_dim, num_layers=a.num_layers, type_dim=a.type_dim,
        contrastive_epochs=a.contrastive_epochs, query_epochs=a.query_epochs,
        lr=a.lr, seed=a.seed, cuda=a.cuda, device=a.device,
        stage_a_n_neg_sample=a.stage_a_n_neg_sample,
        stage_a_xgraph_queue=a.stage_a_xgraph_queue,
        resume_from=a.resume_from,
        checkpoint_every_epochs=a.checkpoint_every_epochs,
        throughput_log_every=a.throughput_log_every,
        log_every=a.log_every,
        tangent_scale=a.tangent_scale,
        radial_reg_weight=a.radial_reg_weight,
        radial_reg_weight_end=a.radial_reg_weight_end,
        temperature=a.temperature, anchors_per_step=a.anchors_per_step,
        positive_mix=a.positive_mix, use_tangent_approx=a.use_tangent_approx,
        neighbor_exclude_k=a.neighbor_exclude_k, margin=a.margin,
        stage_b_loss=a.stage_b_loss,
        stage_b_n_pairs=a.stage_b_n_pairs,
        stage_b_pos_threshold=a.stage_b_pos_threshold,
        stage_b_neg_threshold=a.stage_b_neg_threshold,
        stage_b_negatives=a.stage_b_negatives,
        stage_b_n_positives=a.stage_b_n_positives,
        stage_b_temperature=a.stage_b_temperature,
        train_frac=a.train_frac,
        train_graphs_frac=a.train_graphs_frac,
        graphs_split_seed=a.graphs_split_seed,
        uniformity_reg_weight=a.uniformity_reg_weight,
        uniformity_t=a.uniformity_t,
        temporal_aux_weight=a.temporal_aux_weight,
        stage_b_head=a.stage_b_head,
        include_intrinsic=not a.no_intrinsic, curvature=a.curvature,
        query_head_arch=a.query_head_arch,
        query_head_norm=a.query_head_norm,
        lr_query=a.lr_query,
        skip_stage_a=a.skip_stage_a,
        load_encoder=a.load_encoder,
        assert_encoder_sha=a.assert_encoder_sha,
        attn_type_table=a.attn_type_table,
        freeze_mode=a.freeze_mode,
        stage_c_epochs=a.stage_c_epochs,
        lr_encoder=a.lr_encoder,
        edge_preserve_weight=a.edge_preserve_weight,
        radius_stability_weight=a.radius_stability_weight,
        baseline_encoder=a.baseline_encoder,
        no_autocast=a.no_autocast,
        early_stop=a.early_stop,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
