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
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

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
from src.modelsv3.ranking import listwise_ranking_loss, pairwise_ranking_loss
from src.modelsv2.layers import poincare_ops as P
from src.training.metrics import MetricAccumulator


NODE_TYPE_SLICE = slice(0, 12)


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
    train_frac: float = 1.0
    train_graphs_frac: float = 1.0
    graphs_split_seed: int = 0
    uniformity_reg_weight: float = 0.0
    uniformity_t: float = 2.0
    include_intrinsic: bool = True
    curvature: float = 1.0


def _build_encoder(cfg: Config, dataset: CorpusDataset):
    num_edge_types = dataset.num_edge_types_max
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


def _stage_a(cfg: Config, model, dataset: CorpusDataset, device) -> list[dict]:
    """Contrastive pretraining. Iterates over all samples in the dataset
    (ignoring their labels / queries); each sample yields one graph, one
    positive-sampler step."""
    hyperbolic = cfg.model == "hyperbolic"
    opt, opt_name = _make_optimizer(model, cfg.lr, hyperbolic=hyperbolic)
    print(f"[stage-A] optimizer: {opt_name}  |  lr={cfg.lr}")

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

    history: list[dict] = []
    step = 0
    for epoch in range(cfg.contrastive_epochs):
        order = rng.permutation(steps_per_epoch)
        for batch_idx in order:
            sample: Sample = dataset[int(batch_idx)]
            sample = _sample_to_device(sample, device)
            graph_idx = int(dataset.index[int(batch_idx)][0])
            sampler = get_sampler(graph_idx, sample.x, sample.edge_index)
            batch = sampler.sample(cfg.anchors_per_step)

            node_emb = _encode(model, sample)
            loss_info, diag = poincare_infonce(
                node_emb=node_emb,
                anchor_idx=torch.from_numpy(batch.anchor_idx).to(device),
                positive_idx=torch.from_numpy(batch.positive_idx).to(device),
                valid_mask=torch.from_numpy(batch.valid_mask).to(device),
                c=getattr(model, "c", torch.tensor(cfg.curvature)),
                temperature=cfg.temperature,
                use_tangent_approx=cfg.use_tangent_approx,
            )
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

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    f"unif={unif_val:+.4f}"
                )
            history.append(
                {"step": step, "epoch": epoch, **diag,
                 "reg_w": reg_w, "uniformity": unif_val}
            )
            step += 1

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

    opt = torch.optim.Adam(query_encoder.parameters(), lr=cfg.lr)
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
            q_point = query_encoder(sample.query)
            if cfg.stage_b_loss == "pairwise":
                c_val = getattr(model, "c", torch.tensor(cfg.curvature))
                loss, diag = pairwise_ranking_loss(
                    query_point=q_point,
                    node_embeddings=node_emb,
                    labels=sample.labels,
                    c=c_val,
                    margin=cfg.margin,
                    n_pairs=16,
                    euclidean=euclidean,
                )
            elif cfg.stage_b_loss == "listwise":
                c_val = getattr(model, "c", torch.tensor(cfg.curvature))
                loss, diag = listwise_ranking_loss(
                    query_point=q_point,
                    node_embeddings=node_emb,
                    labels=sample.labels,
                    c=c_val,
                    temperature=cfg.temperature,
                    euclidean=euclidean,
                )
            else:
                raise ValueError(f"unknown stage_b_loss: {cfg.stage_b_loss!r}")

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

    stage_a_history = _stage_a(cfg, encoder, train_ds, device)

    query_encoder = QueryToBall(
        query_dim=train_ds.query_dim,
        hidden_dim=cfg.hidden_dim,
        c=cfg.curvature,
        euclidean=(cfg.model == "euclidean"),
    ).to(device)
    q_params = sum(p.numel() for p in query_encoder.parameters() if p.requires_grad)
    print(f"[train_v3] query encoder params: {q_params}")

    stage_b_history = _stage_b(cfg, encoder, query_encoder, train_ds, device)

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
            "stage_a": cfg.contrastive_epochs,
            "stage_b": cfg.query_epochs,
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
    torch.save(encoder.state_dict(), out_dir / "encoder.pt")
    torch.save(query_encoder.state_dict(), out_dir / "query_encoder.pt")

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
                   choices=["pairwise", "listwise"])
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
    p.add_argument("--no-intrinsic", action="store_true",
                   help="Skip the per-run intrinsic probe (silhouette, nn@5).")
    p.add_argument("--curvature", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    cfg = Config(
        corpus=a.corpus, task=a.task, model=a.model, out=a.out,
        hidden_dim=a.hidden_dim, num_layers=a.num_layers, type_dim=a.type_dim,
        contrastive_epochs=a.contrastive_epochs, query_epochs=a.query_epochs,
        lr=a.lr, seed=a.seed, cuda=a.cuda, log_every=a.log_every,
        tangent_scale=a.tangent_scale,
        radial_reg_weight=a.radial_reg_weight,
        radial_reg_weight_end=a.radial_reg_weight_end,
        temperature=a.temperature, anchors_per_step=a.anchors_per_step,
        positive_mix=a.positive_mix, use_tangent_approx=a.use_tangent_approx,
        neighbor_exclude_k=a.neighbor_exclude_k, margin=a.margin,
        stage_b_loss=a.stage_b_loss, train_frac=a.train_frac,
        train_graphs_frac=a.train_graphs_frac,
        graphs_split_seed=a.graphs_split_seed,
        uniformity_reg_weight=a.uniformity_reg_weight,
        uniformity_t=a.uniformity_t,
        include_intrinsic=not a.no_intrinsic, curvature=a.curvature,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
