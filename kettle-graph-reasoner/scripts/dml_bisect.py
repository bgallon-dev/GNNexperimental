"""Time-boxed DML backward-failure bisector.

Stage A's ``total.backward()`` fails on DML. The earlier smoke only
exercised the *head's* backward on a detached encoder output, so it
missed this. This script climbs progressively more of Stage A's actual
graph, runs ``backward()`` after each layer, and prints which step is
the first to fail. Output names the failing component so we can either
patch / replace the offending op or pivot to Colab.

Steps, smallest to largest:
  S1 encoder forward only (no backward) -> baseline (already known green)
  S2 encoder forward + scalar loss = mean(||emb||^2) + backward -> tests
     gradient flow through the full encoder (no InfoNCE / no Riemannian
     optim involved)
  S3 + poincare_infonce loss + plain Adam backward
  S4 + RiemannianAdam over the encoder's ManifoldParameter(s), if any
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

import torch_directml as tdml  # type: ignore

from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv3.contrastive import PositiveSampler, poincare_infonce
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3


def _label(s: str) -> None:
    print(f"\n----- {s} -----", flush=True)


def main() -> int:
    dev = tdml.device()
    print(f"device={dev} torch={torch.__version__}")
    # Smallest repo: requests at 6059 nodes (vs pydantic at 62k).
    npz = sorted(Path("src/data/corpus/code_v1").glob("*requests*.npz"))[0]
    print(f"graph: {npz.name}")
    with np.load(npz) as z:
        g = _build_graph_tensors(z)
        x_np = z["x"]
        ei_np = z["edge_index"]

    # Fresh encoder, all default flags except small-gain + tangent_scale.
    enc = KettleGraphReasonerV3(
        node_feat_dim=32, edge_feat_dim=13,
        hidden_dim=64, num_layers=2, type_dim=8, c=1.0,
        num_edge_types_max=30, node_feat_dim_schema=4,
        tangent_scale_init=0.1,
    ).to(dev)
    enc.train()

    def fwd():
        return enc(
            g["x"].to(dev), g["edge_index"].to(dev), g["edge_type"].to(dev),
            g["edge_descriptor"].to(dev),
            node_descriptor=g["node_descriptor"].to(dev),
        ).node_embeddings

    # ---- S2: trivial scalar loss + backward through encoder ----
    _label("S2: encoder backward via scalar loss mean(||h||^2)")
    try:
        emb = fwd()
        loss = (emb * emb).sum() / emb.shape[0]
        loss.backward()
        print(f"  OK  loss={float(loss):.4f}  "
              f"grad-mean={float(next(p for p in enc.parameters() if p.grad is not None).grad.abs().mean()):.2e}")
    except Exception as e:  # noqa
        print(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\nVERDICT: encoder-internal backward op fails on DML.")
        return 2
    enc.zero_grad()

    # ---- S3: poincare_infonce loss + Adam backward ----
    _label("S3: encoder backward via poincare_infonce + Adam")
    try:
        sampler = PositiveSampler(x_np, ei_np, neighbor_exclude_k=1)
        batch = sampler.sample(n_anchors=32)
        emb = fwd()
        loss, _ = poincare_infonce(
            node_emb=emb,
            anchor_idx=torch.from_numpy(batch.anchor_idx).to(dev),
            positive_idx=torch.from_numpy(batch.positive_idx).to(dev),
            valid_mask=torch.from_numpy(batch.valid_mask).to(dev),
            c=enc.c, temperature=1.0,
        )
        opt = torch.optim.Adam(enc.parameters(), lr=1e-4)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"  OK  loss={float(loss):.4f}")
    except Exception as e:  # noqa
        print(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\nVERDICT: InfoNCE backward fails on DML "
              "(encoder backward via scalar loss DID work).")
        return 3
    enc.zero_grad()

    # ---- S4: RiemannianAdam (geoopt) ----
    _label("S4: RiemannianAdam step on encoder params")
    try:
        import geoopt
        opt = geoopt.optim.RiemannianAdam(enc.parameters(), lr=1e-4)
        emb = fwd()
        loss = (emb * emb).sum() / emb.shape[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"  OK  RiemannianAdam step ok loss={float(loss):.4f}")
    except Exception as e:  # noqa
        print(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\nVERDICT: RiemannianAdam over encoder params fails on DML "
              "(Adam over same params would be the workaround).")
        return 4

    print("\nALL GREEN: DML can train the encoder; sweep should run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
