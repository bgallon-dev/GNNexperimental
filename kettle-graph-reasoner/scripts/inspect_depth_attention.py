r"""E3 follow-up: does the frozen l4 encoder's DepthAttention actually
read the deep (L2/L3) snapshots — and does that differ by domain?

Motivation (Docs/ARCH_EFFICIENCY_PLAN.md E3, landed verdict): depth-2
passes archival ball-order (0.8845 ~ 0.8852) but FAILS the cross-domain
seeded mixture lens (0.8177 vs 0.8353). If the final depth-attention mix
puts real softmax mass on L2/L3 — more so on code-graph inputs — that
explains why removing those layers is archival-inert yet mixture-hostile.

Method: replicate the encoder forward loop, capture the tangent
snapshots, and compute the final-aggregation alpha (query_idx = L-1)
exactly as DepthAttention.forward does. Report mean (and p10/p90 over
nodes) of per-layer mass, per domain:
  A) real archival all6 graphs (first 20)
  B) the tutorstructure code graph (the E3-failing domain)

    PYTHONIOENCODING=utf-8 py -m scripts.inspect_depth_attention
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.codegraph.harness import _build_encoder
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv2.layers import poincare_ops as P

CKPT = "frozen/kgr-v1.0-2026-07-07/encoder_baseline"
CORPUS = "src/data/corpus/real_domain_eval_all6"
CODE_NPZ = "runs/blend_mixture_mvpA/graph_tutorstructure_patch.npz"
OUT = Path("runs/inspect_depth_attention")
N_ARCHIVAL = 20


@torch.no_grad()
def _alphas(enc, g, device) -> torch.Tensor:
    """(L, N) final-aggregation depth-attention weights."""
    c = enc.c
    h_tan = enc.node_in(g["x"].to(device)) * enc.tangent_scale
    h = P.expmap0(h_tan, c)
    edge_type_emb, _ = enc.schema_encoder(
        g["edge_descriptor"].to(device), g["node_descriptor"].to(device))
    ei = g["edge_index"].to(device)
    et = g["edge_type"].to(device)
    snaps = []
    for attn, mp in zip(enc.attn_layers, enc.mp_layers):
        alpha = attn(h, ei, et, type_emb_override=edge_type_emb)
        h = mp(h, ei, edge_weight=alpha)
        snaps.append(P.logmap0(h, c))
    da = enc.depth_attention
    stack = torch.stack(snaps, dim=0)                    # (L, N, D)
    keys = da.key_norm(stack)
    q = da.depth_queries[enc.num_layers - 1]
    logits = torch.einsum("d,lnd->ln", q, keys)
    return logits.softmax(dim=0)                         # (L, N)


def _summ(alphas_list) -> dict:
    a = torch.cat(alphas_list, dim=1)                    # (L, sum N)
    out = {}
    for l_idx in range(a.shape[0]):
        v = a[l_idx].numpy()
        out[f"L{l_idx}"] = {
            "mean": float(v.mean()),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
        }
    out["deep_mass_mean_L2plus"] = float(a[2:].sum(0).mean()) \
        if a.shape[0] > 2 else 0.0
    out["n_nodes"] = int(a.shape[1])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    enc = None
    arch = []
    for f in sorted(Path(CORPUS).glob("graph_*.npz"))[:N_ARCHIVAL]:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(CKPT), g, device)
        arch.append(_alphas(enc, g, device))
    code = []
    with np.load(CODE_NPZ, allow_pickle=True) as z:
        code.append(_alphas(enc, _build_graph_tensors(z), device))

    report = {"archival_all6": _summ(arch), "tutorstructure": _summ(code)}
    print("final depth-attention mass per layer (mean [p10, p90]):")
    for dom, s in report.items():
        line = "  ".join(
            f"L{i}={s[f'L{i}']['mean']:.3f}[{s[f'L{i}']['p10']:.2f},"
            f"{s[f'L{i}']['p90']:.2f}]"
            for i in range(enc.num_layers))
        print(f"{dom:<16} {line}  deep(L2+)={s['deep_mass_mean_L2plus']:.3f} "
              f"(n={s['n_nodes']})")
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    print(f"report: {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
