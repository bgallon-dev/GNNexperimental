"""DirectML compatibility smoke for the KGR stack.

Exercises the exact ops a scaling sweep needs:
  1. torch-directml device alloc + dtype
  2. KettleGraphReasonerV3 forward on a real tier1-schema graph (load
     the shipped ckpt) -> node embeddings on the ball
  3. expmap0 / logmap0 / mobius math on DML tensors
  4. QueryToBall forward + pairwise_ranking_loss backward (the Stage-B
     loop that trains the head)
  5. Riemannian Adam from geoopt over a small parameter (Stage-A
     optimizer; verifies geoopt + DML interop)

Exit non-zero on any failure; we want a single clear green/red signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import torch_directml as tdml  # type: ignore

from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv3.distance_scoring import score_from_embeddings
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3
from src.modelsv3.query_encoder import QueryToBall
from src.modelsv3.ranking import pairwise_ranking_loss
from src.modelsv2.layers import poincare_ops as P


def main() -> int:
    print("[1] torch-directml device")
    dev = tdml.device()
    print(f"    device={dev}  torch={torch.__version__}")
    a = torch.randn(8, 16, device=dev)
    b = torch.tanh(a @ a.t())
    print(f"    matmul+tanh ok, |b|={float(b.abs().mean()):.3f}")

    print("[2] load tier1 graph + frozen encoder forward on DML")
    npz_path = Path("runs/codegraph_cv/graph_03_tutorstructure_patch.npz")
    if not npz_path.exists():
        # any per-repo NPZ from the LORO-CV run works
        cands = sorted(Path("runs/codegraph_cv").glob("graph_*.npz"))
        if not cands:
            print(f"    no NPZ under runs/codegraph_cv/ -- run the harness first")
            return 2
        npz_path = cands[0]
    with np.load(npz_path) as z:
        g = _build_graph_tensors(z)
    import json
    ck = Path("runs/sweep_arch_hyp/h128_l4_seed1")
    cfg = json.load(open(ck / "summary.json"))["config"]
    enc = KettleGraphReasonerV3(
        node_feat_dim=g["x"].shape[1],
        edge_feat_dim=g["edge_descriptor"].shape[1],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        type_dim=cfg["type_dim"],
        c=cfg["curvature"],
        num_edge_types_max=g["edge_descriptor"].shape[0],
        node_feat_dim_schema=g["node_descriptor"].shape[1],
        tangent_scale_init=cfg["tangent_scale"],
    )
    enc.load_state_dict(torch.load(ck / "encoder.pt", map_location="cpu"))
    enc.to(dev).eval()
    with torch.no_grad():
        out = enc(
            g["x"].to(dev), g["edge_index"].to(dev), g["edge_type"].to(dev),
            g["edge_descriptor"].to(dev),
            node_descriptor=g["node_descriptor"].to(dev),
        )
    emb = out.node_embeddings
    hn = float(emb.norm(dim=1).mean())
    print(f"    emb {tuple(emb.shape)}  |h|_mean={hn:.3f}  (CPU reference ~0.87)")

    print("[3] expmap0 / logmap0 / Mobius round-trip on DML")
    v = torch.randn(64, cfg["hidden_dim"], device=dev) * 0.1
    p = P.expmap0(v, enc.c)
    v2 = P.logmap0(p, enc.c)
    err = float((v - v2).abs().mean())
    print(f"    expmap0/logmap0 round-trip mean-abs err={err:.2e}  (want < 1e-5)")
    if err > 1e-3:
        print("    !!! round-trip error too large; DML may be lowering ops imprecisely")
        return 3

    print("[4] QueryToBall + pairwise_ranking_loss backward on DML")
    head = QueryToBall(
        query_dim=18, hidden_dim=cfg["hidden_dim"], c=cfg["curvature"],
        euclidean=False, arch="qh0",
    ).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    q = torch.randn(18, device=dev)
    cand_emb = emb[:64].detach()
    lab = torch.zeros(64, device=dev)
    lab[:5] = 1.0
    qp = head(q)
    loss, diag = pairwise_ranking_loss(qp, cand_emb, lab, c=enc.c, margin=0.5)
    loss.backward()
    opt.step()
    print(f"    loss={float(loss):.4f}  rank_acc={diag.get('rank_accuracy', 0):.3f}  step ok")

    print("[5] geoopt Riemannian Adam on DML (Stage-A path)")
    try:
        import geoopt
        ball = geoopt.PoincareBall(c=enc.c.item())
        # 4 anchor points; train them to push apart slightly.
        pts = geoopt.ManifoldParameter(
            P.expmap0(torch.randn(4, cfg["hidden_dim"], device=dev) * 0.05, enc.c),
            manifold=ball,
        )
        ropt = geoopt.optim.RiemannianAdam([pts], lr=1e-3)
        ropt.zero_grad()
        dists = ball.dist(pts.unsqueeze(0), pts.unsqueeze(1))  # (4,4)
        loss = -(dists.sum())  # want them far apart
        loss.backward()
        ropt.step()
        print(f"    RiemannianAdam step ok loss={float(loss):.3f}")
    except Exception as e:  # noqa
        print(f"    !!! geoopt step failed on DML: {type(e).__name__}: {e}")
        print("    Stage-A optimizer would need a fallback (plain Adam in tangent space)")
        return 4

    print("\nGREEN: DirectML stack works end-to-end for the sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
