r"""Task-classification head for v0.2 classification tasks.

Wraps the existing ``QueryToBall`` body (in euclidean / tangent-space
mode — we don't need an ``expmap0`` since the readout is into label
space, not onto the manifold) and adds a small ``nn.Linear`` to produce
class logits.

One instance per classification task; the head is trained alongside
the encoder-frozen pipeline exactly like the ranking ``QueryToBall``,
but consumes the case's ``query_vec`` only (encoder embeddings are
*not* read — classification is a function of the query alone, learning
``(query_anchor, query_anchor2_if_paired) -> label`` directly).

Small-gain Xavier init on the linear readout matches the boundary-
saturation mitigation recipe used everywhere else in the project.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..modelsv3.query_encoder import QueryToBall


class TaskClassifierHead(nn.Module):
    """``QueryToBall(euclidean=True)`` body + linear readout to logits."""

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        n_labels: int,
        c: float = 1.0,
        arch: str = "qh0",
    ) -> None:
        super().__init__()
        # The inner body produces a tangent-space vector (no expmap0)
        # because we set ``euclidean=True``. Reusing QueryToBall here
        # gives us the same small-gain init regime and the same
        # backward-graph shape the rest of the harness expects.
        self.body = QueryToBall(
            query_dim=query_dim,
            hidden_dim=hidden_dim,
            c=c,
            euclidean=True,
            arch=arch,
        )
        self.classifier = nn.Linear(hidden_dim, n_labels)
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.05)
        nn.init.zeros_(self.classifier.bias)
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.n_labels = n_labels

    def forward(self, query: Tensor) -> Tensor:
        """``query``: ``(query_dim,)`` or ``(B, query_dim)``. Returns
        per-class logits, shape ``(n_labels,)`` or ``(B, n_labels)``."""
        h = self.body(query)
        return self.classifier(h)
