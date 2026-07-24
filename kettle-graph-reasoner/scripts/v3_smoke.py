r"""MVP smoke test for KGR v3 — single graph, 100 gradient steps.

Gate between "we have the pieces written" and "we build out the full
apparatus". Loads one NPZ from tier1, runs 100 InfoNCE + radial-reg steps
against the v3 encoder, and prints diagnostics so we can eyeball whether
the core mechanic works before committing to more code.

Pass criteria (eyeballed — not a pytest assertion):
    - Loss trends downward (not strictly monotone; noise is OK).
    - |h|_mean stays in [0.2, 0.6]; |h|_max below 0.95.
    - Mean positive sim rises above mean negative sim by step ~50.
    - Effective negatives per anchor > ~30.
    - No NaN/Inf in loss, gradients, or embeddings.

Usage (from kettle-graph-reasoner/ root):
    python scripts/v3_smoke.py [path/to/graph.npz]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Make `src.` importable when running from the kettle-graph-reasoner/ root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import _build_graph_tensors  # noqa: E402
from src.modelsv3.contrastive import (  # noqa: E402
    PositiveSampler,
    poincare_infonce,
)
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3  # noqa: E402


def main(npz_path: str | None = None) -> int:
    npz_path = npz_path or "src/data/corpus/tier1/graph_000000.npz"
    print(f"[smoke] loading {npz_path}")

    with np.load(npz_path) as npz:
        graph = _build_graph_tensors(npz)
        x_np = npz["x"].astype(np.float32)
        edge_index_np = npz["edge_index"].astype(np.int64)

    x = graph["x"]
    edge_index = graph["edge_index"]
    edge_type = graph["edge_type"]
    edge_descriptor = graph["edge_descriptor"]
    node_descriptor = graph["node_descriptor"]
    N = x.size(0)
    E = edge_index.size(1)
    print(f"[smoke] graph: N={N}, E={E}")

    torch.manual_seed(0)
    np.random.seed(0)

    model = KettleGraphReasonerV3(
        node_feat_dim=x.size(1),
        edge_feat_dim=edge_descriptor.size(-1),
        hidden_dim=32,
        num_layers=3,
        type_dim=8,
        c=1.0,
        num_edge_types_max=int(edge_type.max().item()) + 1 + 5,  # slack
        node_feat_dim_schema=node_descriptor.size(-1),
    )
    n_params = model.parameter_count()
    print(f"[smoke] v3 model parameters: {n_params}")

    sampler = PositiveSampler(
        x=x_np,
        edge_index=edge_index_np,
        neighbor_exclude_k=1,
        edge_fraction=0.5,
        low_cos_threshold=0.4,
        rng=np.random.default_rng(0),
    )

    try:
        from geoopt.optim import RiemannianAdam
        opt = RiemannianAdam(model.parameters(), lr=1e-3)
        print("[smoke] optimizer: geoopt.RiemannianAdam")
    except ImportError:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        print("[smoke] optimizer: torch.optim.Adam (geoopt unavailable)")

    # Radial-reg decay from 0.01 to 0.001 over 100 steps (linear), mirroring
    # the train.py recipe's decay-to-small-positive floor.
    reg_start, reg_end, n_steps = 0.01, 0.001, 100
    anchors_per_step = min(64, N)
    temperature = 1.0

    print(
        "[smoke] starting training: "
        f"steps={n_steps}, anchors={anchors_per_step}, tau={temperature}, "
        f"reg {reg_start}->{reg_end}"
    )

    history = []
    for step in range(n_steps):
        batch = sampler.sample(anchors_per_step)
        anchor_idx = torch.from_numpy(batch.anchor_idx)
        positive_idx = torch.from_numpy(batch.positive_idx)
        valid_mask = torch.from_numpy(batch.valid_mask)

        out = model(
            x, edge_index, edge_type, edge_descriptor,
            node_descriptor=node_descriptor,
        )
        loss, diag = poincare_infonce(
            node_emb=out.node_embeddings,
            anchor_idx=anchor_idx,
            positive_idx=positive_idx,
            valid_mask=valid_mask,
            c=model.c,
            temperature=temperature,
        )
        frac = step / max(n_steps - 1, 1)
        reg_w = reg_start + frac * (reg_end - reg_start)
        radial = (out.node_embeddings.norm(dim=-1, p=2) ** 2).mean()
        total = loss + reg_w * radial

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if not torch.isfinite(total):
            print(f"[smoke] FAIL: non-finite loss at step {step} — total={total.item()}")
            return 1

        history.append({
            "step": step,
            "loss": diag["loss"],
            "reg": reg_w * float(radial.item()),
            "pos_sim": diag["mean_pos_sim"],
            "neg_sim": diag["mean_neg_sim"],
            "eff_negs": diag["eff_negs_per_anchor"],
            "h_mean": diag["mean_h_norm"],
            "h_max": diag["max_h_norm"],
        })
        if step % 10 == 0 or step == n_steps - 1:
            print(
                f"[smoke] step {step:3d}  "
                f"loss={diag['loss']:.4f}  "
                f"pos={diag['mean_pos_sim']:+.3f}  "
                f"neg={diag['mean_neg_sim']:+.3f}  "
                f"gap={diag['mean_pos_sim'] - diag['mean_neg_sim']:+.3f}  "
                f"eff_negs={diag['eff_negs_per_anchor']:.0f}  "
                f"|h|mean={diag['mean_h_norm']:.3f}  "
                f"|h|max={diag['max_h_norm']:.3f}  "
                f"reg={reg_w * float(radial.item()):.4f}"
            )

    # Eyeball summary
    first = history[:20]
    last = history[-20:]
    avg = lambda lst, k: sum(d[k] for d in lst) / len(lst)
    print()
    print("[smoke] === summary ===")
    print(f"[smoke] loss first-20 mean = {avg(first, 'loss'):.4f}")
    print(f"[smoke] loss last-20  mean = {avg(last, 'loss'):.4f}  "
          f"(want: lower)")
    print(f"[smoke] pos-neg gap first-20 mean = {avg(first, 'pos_sim') - avg(first, 'neg_sim'):+.3f}")
    print(f"[smoke] pos-neg gap last-20  mean = {avg(last, 'pos_sim') - avg(last, 'neg_sim'):+.3f}  "
          f"(want: positive and growing)")
    print(f"[smoke] |h|_mean last-20 mean = {avg(last, 'h_mean'):.3f}  (want: 0.2–0.6)")
    print(f"[smoke] |h|_max  last-20 mean = {avg(last, 'h_max'):.3f}  (want: < 0.95)")
    print(f"[smoke] eff_negs last-20 mean = {avg(last, 'eff_negs'):.0f}")

    # Hard sanity: positive sim MUST exceed negative sim after training
    # Otherwise the contrastive signal is not being learned.
    final_gap = avg(last, 'pos_sim') - avg(last, 'neg_sim')
    if final_gap <= 0:
        print(f"[smoke] WARN: final pos-neg gap = {final_gap:+.3f} ≤ 0 — "
              "contrastive signal not learning")
        return 2

    print("[smoke] OK: smoke criteria met")
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(main(arg))
