"""
scale_stress.py -- "how big does KGR go before the laptop dies, for fun".

NOTE: this deliberately violates architectural commitment #5 (tiny-by-design,
500K-2M params). It is a sandbox stress run, not a change to the real model.
The on-thesis payoff: CLAUDE.md Known-Issues predicts large hidden_dim
re-triggers Poincare boundary saturation (tangent vectors into expmap0 have
norm ~sqrt(hidden_dim); the gain=0.05 / tangent_scale=0.1 recipe was tuned at
hidden_dim ~64-128). So as we climb the param ladder we also watch ||h||
mean/max -- does the tiny-design numeric mitigation break at scale?

Graph is held tiny+fixed so memory is PARAMETER-dominated, not activation-
dominated -- we're measuring model size, not batch size. Each rung:
instantiate -> param count -> ||h|| at init (no_grad) -> one forward+backward
(loss = mean ||h||) -> peak RSS. A rung whose projected fwd+bwd footprint
exceeds ~75% of currently-available RAM is SKIPPED, not attempted -- pushing
to the edge without OOM-killing the user's box.
"""
from __future__ import annotations

import sys
import time
import gc
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3  # noqa: E402

# Fixed tiny synthetic graph -- param-dominated regime
N, E = 400, 1600
NODE_FEAT, EDGE_FEAT, NODE_FEAT_SCHEMA = 16, 8, 8
N_ETYPES, N_NTYPES = 12, 10
PROJ_DEFAULT, TINY_BUDGET = 0.5e6, 2.0e6   # commitment #5 reference band

torch.manual_seed(0)
_g = {
    "x": torch.randn(N, NODE_FEAT),
    "ei": torch.randint(0, N, (2, E)),
    "et": torch.randint(0, N_ETYPES, (E,)),
    "ed": torch.randn(N_ETYPES, EDGE_FEAT),
    "nd": torch.randn(N_NTYPES, NODE_FEAT_SCHEMA),
}


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def run_rung(hidden_dim: int, num_layers: int) -> dict:
    avail = psutil.virtual_memory().available / 1e9
    t0 = time.monotonic()
    model = KettleGraphReasonerV3(
        node_feat_dim=NODE_FEAT, edge_feat_dim=EDGE_FEAT,
        hidden_dim=hidden_dim, num_layers=num_layers,
        type_dim=8, c=1.0, num_edge_types_max=N_ETYPES,
        node_feat_dim_schema=NODE_FEAT_SCHEMA,
    )
    p = model.parameter_count()
    # init ||h|| (boundary-saturation diagnostic, CLAUDE.md Known-Issues)
    with torch.no_grad():
        out = model(_g["x"], _g["ei"], _g["et"], _g["ed"], _g["nd"])
        h = out.node_embeddings
        hn = h.norm(dim=-1)
        hmean_i, hmax_i = float(hn.mean()), float(hn.max())
    # one fwd+bwd
    out = model(_g["x"], _g["ei"], _g["et"], _g["ed"], _g["nd"])
    loss = out.node_embeddings.norm(dim=-1).mean()
    loss.backward()
    gnorm = torch.sqrt(sum((q.grad.detach() ** 2).sum()
                           for q in model.parameters()
                           if q.grad is not None)).item()
    rss = _rss_gb()
    secs = time.monotonic() - t0
    finite = bool(torch.isfinite(loss).item()) and (gnorm == gnorm)
    del model, out, loss
    gc.collect()
    return dict(hidden=hidden_dim, layers=num_layers, params=p,
                x_over_tiny=p / TINY_BUDGET, h_mean=hmean_i, h_max=hmax_i,
                loss_finite=finite, gnorm=gnorm, rss_gb=rss,
                avail_gb=avail, secs=secs)


def main() -> int:
    # (hidden_dim, num_layers) ladder: widen first, then go deep
    ladder = [
        (64, 3),     # ~ project / tiny-by-design reference
        (256, 3), (512, 3), (1024, 3), (2048, 3),
        (4096, 3), (6144, 3), (8192, 3),
        (1024, 8), (2048, 8),     # deep variants
    ]
    print(f"box: {psutil.virtual_memory().total/1e9:.1f}GB total, "
          f"{psutil.virtual_memory().available/1e9:.1f}GB free | "
          f"tiny-by-design band {PROJ_DEFAULT/1e6:.1f}-{TINY_BUDGET/1e6:.0f}M "
          f"params (commitment #5)\n")
    hdr = (f"{'hid':>5} {'L':>2} {'params':>14} {'x>tiny':>8} "
           f"{'|h|_init mean/max':>20} {'loss ok':>7} {'gnorm':>9} "
           f"{'RSS GB':>7} {'s':>5}")
    print(hdr); print("-" * len(hdr))
    ceiling = None
    for hid, L in ladder:
        # crude projection: params*4 (wts) + *4 (grads) + activations/graph
        # slack -> ~*12 bytes; skip if it would blow >75% of free RAM
        try:
            probe = KettleGraphReasonerV3(
                node_feat_dim=NODE_FEAT, edge_feat_dim=EDGE_FEAT,
                hidden_dim=hid, num_layers=L, type_dim=8, c=1.0,
                num_edge_types_max=N_ETYPES,
                node_feat_dim_schema=NODE_FEAT_SCHEMA)
            pp = probe.parameter_count()
            del probe; gc.collect()
        except (MemoryError, RuntimeError) as ex:
            print(f"{hid:>5} {L:>2}  instantiate FAILED: "
                  f"{type(ex).__name__} -> ceiling reached")
            ceiling = ceiling or (hid, L)
            break
        proj_gb = pp * 12 / 1e9
        avail = psutil.virtual_memory().available / 1e9
        if proj_gb > 0.75 * avail:
            print(f"{hid:>5} {L:>2} {pp:>14,} {pp/TINY_BUDGET:>7.0f}x  "
                  f"SKIPPED: proj ~{proj_gb:.1f}GB > 75% of {avail:.1f}GB free")
            ceiling = ceiling or (hid, L)
            continue
        try:
            r = run_rung(hid, L)
        except (MemoryError, RuntimeError) as ex:
            print(f"{hid:>5} {L:>2} {pp:>14,}  fwd/bwd FAILED: "
                  f"{type(ex).__name__}")
            ceiling = ceiling or (hid, L)
            break
        print(f"{r['hidden']:>5} {r['layers']:>2} {r['params']:>14,} "
              f"{r['x_over_tiny']:>7.0f}x "
              f"{r['h_mean']:>9.3f}/{r['h_max']:<9.3f} "
              f"{str(r['loss_finite']):>7} {r['gnorm']:>9.2e} "
              f"{r['rss_gb']:>7.2f} {r['secs']:>5.1f}")
    print()
    print("read: x>tiny = multiples over the 2M tiny-by-design ceiling. "
          "Watch |h|_max -> ~1.0 (=1/sqrt(c)) = Poincare boundary "
          "saturation re-emerging at scale despite the gain=0.05 / "
          "tangent_scale=0.1 mitigation tuned for hidden~64-128.")
    if ceiling:
        print(f"practical ceiling on this box: ~hidden_dim {ceiling[0]} "
              f"(num_layers {ceiling[1]}) before RAM gives out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
