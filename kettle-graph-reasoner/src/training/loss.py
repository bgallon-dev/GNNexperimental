r"""Relevance loss for KGR training.

The model outputs per-node and per-edge scores already pushed through a
sigmoid, so we use the *functional* ``binary_cross_entropy`` (not
``_with_logits``). Loss type is chosen per task:

    task 0 (provenance chain traversal)   — BCE  (labels are 0/1-ish with 1/d decay on intermediates)
    task 1 (entity resolution)            — BCE  (binary participation mask)
    task 2 (temporal scope filtering)     — MSE  (continuous 0/0.5/1 band)
    task 3 (multi-hop relevance decay)    — MSE  (continuous decay score)
    task 4 (subgraph boundary detection)  — BCE  (binary include mask)

Edge targets are the endpoint-average of node labels:
    y_edge[e] = (y_node[src(e)] + y_node[dst(e)]) / 2

This is a deliberate v1 simplification. It undersells edges whose
*structural* role is critical but whose endpoints are only moderately
relevant (the classic provenance-chain edge case). If end-to-end edge
evaluation lags, this is the first thing to revisit — e.g. by marking
edges that lie on the shortest anchor→target path as 1.0 regardless of
endpoint scores.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.hyperbolic_gnn import KGROutput


BCE_TASKS = {0, 1, 4}
MSE_TASKS = {2, 3}


def _pick_loss(task_type: int):
    if task_type in BCE_TASKS:
        return F.binary_cross_entropy
    if task_type in MSE_TASKS:
        return F.mse_loss
    raise ValueError(f"unknown task_type {task_type}")


def relevance_loss(
    output: KGROutput,
    node_labels: Tensor,
    edge_index: Tensor,
    task_type: int,
    edge_weight: float = 0.5,
    eps: float = 1e-6,
    task_weight: float = 1.0,
) -> dict:
    loss_fn = _pick_loss(task_type)

    node_labels = node_labels.clamp(0.0, 1.0)
    node_pred = output.node_scores.clamp(eps, 1.0 - eps)
    node_loss = loss_fn(node_pred, node_labels)

    src, dst = edge_index[0], edge_index[1]
    edge_labels = 0.5 * (node_labels.index_select(0, src) + node_labels.index_select(0, dst))
    edge_pred = output.edge_scores.clamp(eps, 1.0 - eps)
    edge_loss = loss_fn(edge_pred, edge_labels)

    total = task_weight * (node_loss + edge_weight * edge_loss)
    return {
        "loss": total,
        "node_loss": node_loss,
        "edge_loss": edge_loss.detach(),
    }
