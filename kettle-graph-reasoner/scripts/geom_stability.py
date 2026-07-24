"""
geom_stability.py -- long-horizon geometric-stability probe for the v3
hyperbolic encoder.

The open question left by scale_stress.py: init-time numerics are width-robust,
but the documented Poincare boundary saturation is a *many-epoch, layer-
compounding TRAINING* effect that no single fwd+bwd can surface. This runs a
faithful long-horizon Stage-A and instruments the geometry per-epoch AND
per-layer, because the failure may be layer-local rather than global.

Faithfulness is the whole point: the training step is a verbatim reuse of
`train_v3._stage_a`'s inner loop -- same RiemannianAdam, same per-graph
`PositiveSampler`, same `poincare_infonce`, same radial-reg DECAY schedule
(reg_start -> reg_end over the *full* horizon), same grad-clip. We observe
geometry under the real healthy training dynamics, not a trivially-induced
collapse. Stage-A is task-agnostic; --task only scopes the eval-side
detectors.

Per epoch (cheap, fixed held-out val probe graphs, no_grad):
  - ||h|| max / mean / p50 / p99  -- boundary drift, radial inflation
  - ||h|| histogram               -- representational stratification
  - pairwise hyperbolic-dist spread (mean/std/p05/p95) -- compression collapse
  - ALL of the above ALSO per message-passing layer (per_round_embeddings)

Every --eval-every epochs (query-AGNOSTIC by design -- a per-epoch retrained
head would confound encoder drift with head fit):
  - nn_edge_precision@5  -- curvature-sensitive retrieval / bridge / early
                            collapse detector (its drop from peak = the
                            structural "oracle gap over training")
  - silhouette(node-type), nn_label_purity@5 -- geometry-still-matters

Run from scripts/ (repo root importable). Smoke first:
    py geom_stability.py --smoke
Then the real horizon (background):
    py geom_stability.py --epochs 60 --eval-every 5 --out runs/geomstab-hyp-h256-l4-t2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the EXACT canonical objects so the training signal is bit-faithful
# to train_v3._stage_a (identical Config defaults, sampler, loss, decay).
from src.training.train_v3 import (  # noqa: E402
    Config, CorpusDataset, PositiveSampler, poincare_infonce,
    _build_encoder, _make_optimizer, _encode, _sample_to_device,
    silhouette_score, nn_edge_precision_at_k, nn_label_purity_at_k,
    NODE_TYPE_SLICE, P,
)
from torch import nn  # noqa: E402
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3  # noqa: E402


def build_encoder(cfg, ds, fix: str, tan_clamp: float):
    """Mirror train_v3._build_encoder's hyperbolic branch EXACTLY for
    fix in {none,C} (bit-identical model); arm A = A' passes the opt-in
    per_layer_tan_clamp (step-7 tangent-norm clamp); arm D re-inits MP
    weights at gain 0.02."""
    kw = dict(
        node_feat_dim=ds.node_feat_dim,
        edge_feat_dim=ds.edge_feat_dim_schema,
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        type_dim=cfg.type_dim, c=cfg.curvature,
        num_edge_types_max=ds.num_edge_types_max,
        node_feat_dim_schema=ds.node_feat_dim_schema,
        tangent_scale_init=cfg.tangent_scale,
    )
    if fix == "A":
        kw["per_layer_tan_clamp"] = tan_clamp
    enc = KettleGraphReasonerV3(**kw)
    if fix == "D":
        for mp in enc.mp_layers:
            nn.init.xavier_uniform_(mp.weight, gain=0.02)
            if mp.bias is not None:
                nn.init.zeros_(mp.bias)
    return enc

_BINS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                  0.8, 0.9, 0.95, 0.99, 1.0, 1e9])


def _norm_stats(h: torch.Tensor) -> dict:
    n = h.norm(dim=-1, p=2).detach().cpu().numpy()
    hist, _ = np.histogram(n, bins=_BINS)
    return {
        "mean": float(n.mean()), "max": float(n.max()),
        "p50": float(np.percentile(n, 50)),
        "p99": float(np.percentile(n, 99)),
        "hist": (hist / max(1, n.size)).round(4).tolist(),
    }


def _dist_spread(h: torch.Tensor, c, rng: np.random.Generator,
                 m: int = 2000) -> dict:
    ng = h.shape[0]
    if ng < 3:
        return {"mean": 0.0, "std": 0.0, "p05": 0.0, "p95": 0.0}
    a = rng.integers(0, ng, size=m)
    b = rng.integers(0, ng, size=m)
    ok = a != b
    a, b = a[ok], b[ok]
    with torch.no_grad():
        d = P.dist(h[a], h[b], c).detach().cpu().numpy()
    return {"mean": float(d.mean()), "std": float(d.std()),
            "p05": float(np.percentile(d, 5)),
            "p95": float(np.percentile(d, 95))}


def _geom_snapshot(encoder, probes, c, rng) -> dict:
    """Per-epoch geometry on a FIXED probe set: final + per-layer."""
    was = encoder.log_depth
    encoder.log_depth = True
    encoder.eval()
    fin_norm, fin_dist = [], []
    L = encoder.num_layers
    layer_norm = [[] for _ in range(L)]
    layer_dist = [[] for _ in range(L)]
    with torch.no_grad():
        for s in probes:
            out = encoder(s.x, s.edge_index, s.edge_type,
                          s.edge_descriptor, node_descriptor=s.node_descriptor)
            fin_norm.append(_norm_stats(out.node_embeddings))
            fin_dist.append(_dist_spread(out.node_embeddings, c, rng))
            pr = out.per_round_embeddings or []
            for li in range(min(L, len(pr))):
                layer_norm[li].append(_norm_stats(pr[li]))
                layer_dist[li].append(_dist_spread(pr[li], c, rng))
    encoder.log_depth = was

    def _avg(ds, key, sub=None):
        vals = [d[key] if sub is None else d[key] for d in ds]
        if sub is not None:
            return [float(np.mean([d[sub] for d in ds]))]
        return float(np.mean([v for v in vals]))

    def _agg_norm(lst):
        return {k: float(np.mean([d[k] for d in lst]))
                for k in ("mean", "max", "p50", "p99")}

    def _agg_dist(lst):
        return {k: float(np.mean([d[k] for d in lst]))
                for k in ("mean", "std", "p05", "p95")}

    return {
        "final_norm": _agg_norm(fin_norm),
        "final_hist": np.mean([d["hist"] for d in fin_norm], axis=0)
                         .round(4).tolist(),
        "final_dist": _agg_dist(fin_dist),
        "layer_norm": [_agg_norm(layer_norm[li]) for li in range(L)],
        "layer_dist": [_agg_dist(layer_dist[li]) for li in range(L)],
        "layer_hist": [np.mean([d["hist"] for d in layer_norm[li]], axis=0)
                       .round(3).tolist() for li in range(L)],
    }


def _intrinsic(encoder, probes, c) -> dict:
    """Query-AGNOSTIC curvature-sensitive detectors, mean over probe set."""
    encoder.eval()
    sil, nep, nlp = [], [], []
    with torch.no_grad():
        for s in probes:
            ne = _encode(encoder, s)
            tb = s.x[:, NODE_TYPE_SLICE]
            sums = tb.sum(dim=1)
            tl = torch.where(sums > 0, tb.argmax(dim=1),
                             torch.full_like(sums, -1, dtype=torch.long))
            sil.append(silhouette_score(ne.cpu(), tl.cpu(), c=c,
                                        euclidean=False)["mean"])
            nep.append(nn_edge_precision_at_k(ne.cpu(),
                       s.edge_index.cpu(), k=5, c=c,
                       euclidean=False)["mean_precision"])
            nlp.append(nn_label_purity_at_k(ne.cpu(), tl.cpu(), k=5,
                       c=c, euclidean=False)["mean_purity"])
    return {"silhouette": float(np.mean(sil)),
            "nn_edge_prec@5": float(np.mean(nep)),
            "nn_label_purity@5": float(np.mean(nlp))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="src/data/corpus/tier1")
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--probe-n", type=int, default=6)
    ap.add_argument("--out", default="runs/geomstab-hyp-h256-l4-t2")
    ap.add_argument("--fix", choices=["none", "C", "D", "A"], default="none",
                    help="none=baseline; C=intermediate per-round radial-reg; "
                         "D=MP weight gain 0.02 re-init; "
                         "A=opt-in per-layer agg-brake")
    ap.add_argument("--mid-reg-weight", type=float, default=0.01)
    ap.add_argument("--mid-reg-weight-end", type=float, default=0.001)
    ap.add_argument("--tan-clamp", type=float, default=1.1,
                    help="arm A' step-7 tangent-norm cap; ball radius "
                         "capped at tanh(sqrt(c)*tau) (c=1: 1.1->0.80)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.eval_every, a.probe_n = 2, 1, 3

    # Paths are repo-root-relative (train_v3 convention), not cwd-relative.
    repo_root = Path(__file__).resolve().parents[1]
    if not Path(a.corpus).is_absolute():
        a.corpus = str((repo_root / a.corpus).resolve())
    out_dir = Path(a.out)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = Config(corpus=a.corpus, task=a.task, model="hyperbolic",
                 out=str(out_dir), hidden_dim=256, num_layers=4,
                 contrastive_epochs=a.epochs)
    print(f"[geomstab] hidden_dim=256 num_layers=4 task={a.task} "
          f"epochs={a.epochs} fix={a.fix} | faithful Stage-A reuse of "
          f"train_v3._stage_a")

    train_ds = CorpusDataset(corpus_dir=cfg.corpus, split="train",
                             split_seed=0, include_tasks={cfg.task})
    val_ds = CorpusDataset(corpus_dir=cfg.corpus, split="val",
                           split_seed=0, include_tasks={cfg.task})
    encoder = build_encoder(cfg, train_ds, a.fix, a.tan_clamp).to(device)
    if a.fix == "C":
        encoder.log_depth = True  # need per_round in the training graph
    if a.fix == "A":
        import math as _m
        cap = _m.tanh((cfg.curvature ** 0.5) * a.tan_clamp) / (
            cfg.curvature ** 0.5)
        print(f"[geomstab] arm A' (step-7 tan-clamp): tau={a.tan_clamp} "
              f"-> per-layer ball radius capped at {cap:.3f} "
              f"({encoder.num_layers} MP layers, FIXED)")
    if a.fix == "D":
        print("[geomstab] arm D: MP weights re-init xavier gain=0.02")
    if a.fix == "C":
        print(f"[geomstab] arm C: intermediate per-round radial-reg "
              f"{a.mid_reg_weight}->{a.mid_reg_weight_end} (decayed)")
    n_params = sum(p.numel() for p in encoder.parameters()
                   if p.requires_grad)
    in_band = 0.5e6 <= n_params <= 2.0e6
    print(f"[geomstab] encoder params={n_params:,} "
          f"({'within' if in_band else 'OUTSIDE'} tiny-by-design 0.5-2M) | "
          f"train={len(train_ds)} val={len(val_ds)}")

    # Fixed held-out probe set (val split = not trained on) -- same graphs
    # every epoch so trajectories are comparable.
    probes = [_sample_to_device(val_ds[i], device)
              for i in range(min(a.probe_n, len(val_ds)))]
    probe_rng = np.random.default_rng(12345)

    # ---- verbatim _stage_a setup ----
    opt, opt_name = _make_optimizer(encoder, cfg.lr, hyperbolic=True)
    rng = np.random.default_rng(cfg.seed)
    sampler_cache: dict[int, PositiveSampler] = {}

    def get_sampler(gi, x_t, ei_t):
        if gi not in sampler_cache:
            sampler_cache[gi] = PositiveSampler(
                x=x_t.detach().cpu().numpy().astype(np.float32),
                edge_index=ei_t.detach().cpu().numpy().astype(np.int64),
                neighbor_exclude_k=cfg.neighbor_exclude_k,
                edge_fraction=cfg.positive_mix, low_cos_threshold=0.4,
                rng=np.random.default_rng(cfg.seed + 17 + gi))
        return sampler_cache[gi]

    steps_per_epoch = len(train_ds)
    total_steps = steps_per_epoch * cfg.contrastive_epochs
    reg_start = cfg.radial_reg_weight
    reg_end = (reg_start if cfg.radial_reg_weight_end is None
               else cfg.radial_reg_weight_end)
    c = getattr(encoder, "c", torch.tensor(cfg.curvature))
    print(f"[geomstab] opt={opt_name} steps/epoch={steps_per_epoch} "
          f"radial-reg {reg_start}->{reg_end} over {total_steps} steps")

    jsonl = (out_dir / "trajectory.jsonl").open("w", encoding="utf-8")
    peak_nep = -1.0
    drift = {"hmax09_epoch": None, "compress50_epoch": None,
             "nep_drop25_epoch": None, "first_layer_sat": None}
    base_dist = None
    step = 0
    t0 = time.monotonic()
    for epoch in range(cfg.contrastive_epochs):
        encoder.train()
        ep_losses = []
        for bi in rng.permutation(steps_per_epoch):
            s = _sample_to_device(train_ds[int(bi)], device)
            gi = int(train_ds.index[int(bi)][0])
            batch = get_sampler(gi, s.x, s.edge_index).sample(
                cfg.anchors_per_step)
            if a.fix == "C":
                _out = encoder(s.x, s.edge_index, s.edge_type,
                               s.edge_descriptor,
                               node_descriptor=s.node_descriptor)
                node_emb = _out.node_embeddings
                per_round = _out.per_round_embeddings
            else:
                node_emb = _encode(encoder, s)
                per_round = None
            loss_info, diag = poincare_infonce(
                node_emb=node_emb,
                anchor_idx=torch.from_numpy(batch.anchor_idx).to(device),
                positive_idx=torch.from_numpy(batch.positive_idx).to(device),
                valid_mask=torch.from_numpy(batch.valid_mask).to(device),
                c=c, temperature=cfg.temperature,
                use_tangent_approx=cfg.use_tangent_approx)
            frac = step / max(total_steps - 1, 1)
            reg_w = reg_start + frac * (reg_end - reg_start)
            radial = (node_emb.norm(dim=-1, p=2) ** 2).mean()
            total = loss_info + reg_w * radial
            # arm C: decaying radial-reg on the (currently unconstrained)
            # intermediate per-round embeddings -- the CLAUDE.md-stated cause
            # ("only the final layer is regularized").
            if a.fix == "C" and per_round:
                mid_w = (a.mid_reg_weight
                         + frac * (a.mid_reg_weight_end - a.mid_reg_weight))
                mid = torch.stack([(h.norm(dim=-1, p=2) ** 2).mean()
                                   for h in per_round]).mean()
                total = total + mid_w * mid
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss e{epoch} s{step}")
            ep_losses.append(float(diag["loss"]))
            step += 1

        snap = _geom_snapshot(encoder, probes, c, probe_rng)
        rec = {"epoch": epoch, "loss": float(np.mean(ep_losses)),
               "reg_w": reg_w, **snap}
        if epoch % a.eval_every == 0 or epoch == cfg.contrastive_epochs - 1:
            rec["intrinsic"] = _intrinsic(encoder, probes, c)
            nep = rec["intrinsic"]["nn_edge_prec@5"]
            peak_nep = max(peak_nep, nep)
            if (peak_nep > 0 and nep < 0.75 * peak_nep
                    and drift["nep_drop25_epoch"] is None):
                drift["nep_drop25_epoch"] = epoch

        fmax = snap["final_norm"]["max"]
        fdist = snap["final_dist"]["mean"]
        base_dist = base_dist if base_dist is not None else fdist
        if fmax > 0.9 and drift["hmax09_epoch"] is None:
            drift["hmax09_epoch"] = epoch
        if (base_dist > 0 and fdist < 0.5 * base_dist
                and drift["compress50_epoch"] is None):
            drift["compress50_epoch"] = epoch
        for li, ln in enumerate(snap["layer_norm"]):
            if ln["max"] > 0.9 and drift["first_layer_sat"] is None:
                drift["first_layer_sat"] = {"epoch": epoch, "layer": li}

        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        lmax = "/".join(f"{ln['max']:.2f}" for ln in snap["layer_norm"])
        extra = (f" nep@5={rec['intrinsic']['nn_edge_prec@5']:.3f} "
                 f"sil={rec['intrinsic']['silhouette']:+.3f}"
                 if "intrinsic" in rec else "")
        print(f"[e{epoch:>3}] loss={rec['loss']:.4f} "
              f"|h|f mean/max={snap['final_norm']['mean']:.3f}/"
              f"{fmax:.3f} Lmax[{lmax}] "
              f"dist={fdist:.3f}(p05={snap['final_dist']['p05']:.3f}) "
              f"reg={reg_w:.4f}{extra}")

    jsonl.close()
    secs = time.monotonic() - t0
    final_layer_max = [round(ln["max"], 4) for ln in snap["layer_norm"]]
    fix_param = None
    if a.fix == "A":
        import math as _m
        fix_param = {"tau": a.tan_clamp,
                     "radius_cap": round(_m.tanh(
                         (cfg.curvature ** 0.5) * a.tan_clamp)
                         / (cfg.curvature ** 0.5), 4)}
    elif a.fix == "C":
        fix_param = {"mid_reg_weight": a.mid_reg_weight,
                     "mid_reg_weight_end": a.mid_reg_weight_end}
    elif a.fix == "D":
        fix_param = {"mp_weight_gain": 0.02}
    summary = {"fix": a.fix, "params": n_params,
               "within_tiny_band": in_band,
               "epochs": cfg.contrastive_epochs, "task": cfg.task,
               "final_loss": rec["loss"],
               "final_intrinsic": rec.get("intrinsic"),
               "peak_nn_edge_prec@5": peak_nep,
               "final_layer_norm_max": final_layer_max,
               "final_fused_norm": snap["final_norm"],
               "final_dist": snap["final_dist"],
               "fix_param": fix_param,
               "drift": drift, "wall_sec": round(secs, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[geomstab] done in {secs:.0f}s -> {out_dir}/")
    print(f"  drift onsets: {drift}")
    print("  read: hmax->~1.0 = boundary saturation; dist mean collapsing "
          "= compression; nep@5 falling from peak = functional/structural "
          "degradation. first_layer_sat tells you if it is LAYER-LOCAL "
          "(the hypothesis) vs global.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
