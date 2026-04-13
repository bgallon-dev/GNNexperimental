r"""Schema encoder — descriptor-conditioned type embeddings.

Operationalizes Architectural Commitment #3 (schema-portable). Instead of
learning a fixed ``nn.Embedding(num_edge_types, type_dim)`` table tied to
the specific edge-type *IDs* of the training corpus (which would silently
bind the model to a schema), this module learns a *function* from
structural feature descriptors to type embeddings. A new schema supplies
a fresh descriptor matrix; the same learned function produces its type
embeddings. No retraining, no index remapping.

Descriptor contract (not baked into the encoder — the encoder just sees a
feature vector):

    Each edge type is described by a vector of abstract structural
    properties, e.g. ``[is_hierarchical, is_temporal, is_provenance,
    is_symmetric, has_cardinality_hint, ...]``. These are *abstract*
    (applicable to any schema), not domain labels. The data loader fills
    them per-type. The encoder never sees the string "MENTIONS" or
    "DERIVED_FROM" — only the abstract vector. That is what makes learned
    competence transfer.

Node descriptors work the same way, optional.

This module produces ``(T_edge, type_dim)`` tensors meant to *replace*
``EdgeTypedAttention.type_emb(edge_type)`` at forward time. ``EdgeTypedAttention``
has been extended with a ``type_emb_override`` argument for that plumbing.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class SchemaEncoder(nn.Module):
    r"""Two small MLPs that map edge / node structural-feature descriptors
    to type embeddings.

    Parameters
    ----------
    edge_feat_dim : int
        Dimension of the per-edge-type descriptor vector.
    type_dim : int
        Output embedding dimension. Must match the ``type_dim`` of the
        consuming ``EdgeTypedAttention``.
    node_feat_dim : int, optional
        Dimension of the per-node-type descriptor vector. If ``None``, only
        edge embeddings are produced.
    hidden_dim : int
        Width of the hidden layer in both MLPs.
    """

    def __init__(
        self,
        edge_feat_dim: int,
        type_dim: int,
        node_feat_dim: Optional[int] = None,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.node_feat_dim = node_feat_dim
        self.type_dim = type_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, type_dim),
        )
        if node_feat_dim is not None:
            self.node_mlp = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, type_dim),
            )
        else:
            self.node_mlp = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        edge_descriptor: Tensor,
        node_descriptor: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        r"""
        Parameters
        ----------
        edge_descriptor : Tensor, shape ``(T_edge, edge_feat_dim)``
        node_descriptor : Tensor, shape ``(T_node, node_feat_dim)``, optional

        Returns
        -------
        edge_emb : Tensor, shape ``(T_edge, type_dim)``
        node_emb : Tensor or None, shape ``(T_node, type_dim)``
        """
        if edge_descriptor.dim() != 2 or edge_descriptor.size(-1) != self.edge_feat_dim:
            raise ValueError(
                f"edge_descriptor must be (T_edge, {self.edge_feat_dim}); "
                f"got {tuple(edge_descriptor.shape)}"
            )
        edge_emb = self.edge_mlp(edge_descriptor)

        node_emb: Optional[Tensor] = None
        if node_descriptor is not None:
            if self.node_mlp is None:
                raise ValueError(
                    "SchemaEncoder was built without node_feat_dim; cannot "
                    "encode node descriptors"
                )
            if node_descriptor.dim() != 2 or node_descriptor.size(-1) != self.node_feat_dim:
                raise ValueError(
                    f"node_descriptor must be (T_node, {self.node_feat_dim}); "
                    f"got {tuple(node_descriptor.shape)}"
                )
            node_emb = self.node_mlp(node_descriptor)

        return edge_emb, node_emb
