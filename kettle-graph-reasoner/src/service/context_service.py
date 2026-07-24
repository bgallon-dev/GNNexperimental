r"""KGR Context Service — the deployable core.

Turns the frozen KGR encoder's one validated capability — anchor-
conditioned structural ORDERING of a graph neighborhood — into a small,
framework-agnostic service that ranks context nodes for an LLM.

Every design choice is dictated by a measured finding
(Docs/STRESS_TEST_2026-07-10.md, Docs/BENCHMARK_2026-07-07.md):

- ORDER, never a distance-thresholded MASK. A global distance threshold
  does not select a clean subgraph (F1 lift ~0); the ranking is
  near-oracle (ndcg@10 0.885-0.999). So we return a ranked top-k.
- DETERMINISTIC tie-break. Embeddings are permutation-equivariant but the
  ranking is tie-unstable (|dndcg| up to 0.37 under relabeling); we break
  ties on a canonical node id so top-k is reproducible.
- MULTI-ANCHOR via min-distance (== max-score). Compound/union relevant
  sets are not a single neighborhood; a single anchor sits at the random
  floor there, min over 2+ good anchors recovers +0.47. score(node) =
  max over anchors of -dist(node, anchor).
- ANCHOR is load-bearing and the caller's responsibility. A wrong anchor
  collapses the ranking to noise (measured live: precision@10 0.51 -> 0.16
  on a random anchor). CRITICAL, measured 2026-07-10: the score-spread
  statistic does NOT distinguish a correct from a wrong anchor (identical
  ~0.34 for both) -- a wrong anchor yields a confident-looking ranking.
  `score_spread` is exposed as a DESCRIPTIVE stat only; anchor correctness
  CANNOT be self-detected and must be guaranteed upstream.
- STRUCTURAL, not semantic. The encoder rides node topology (random-
  feature message passing), not schema content — so this orders by
  structural relevance to the anchor, and is schema-portable for free.

Core object is `KGRContextService`; `load_graph` embeds a neighborhood
once, then `order_context` answers many queries against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..codegraph.harness import _build_encoder
from ..data.corpus_dataset import _build_graph_tensors
from ..modelsv3.distance_scoring import score_from_embeddings

FROZEN = "frozen/kgr-v1.0-2026-07-07/encoder_baseline"


@dataclass
class ContextItem:
    """One ranked context node handed to the LLM layer."""
    node_id: object          # external id (neo4j id if present, else row)
    row: int                 # row index within the loaded graph
    rank: int                # 1-based
    score: float             # -distance to nearest anchor (higher = closer)
    hop: int | None          # graph hops to nearest anchor (None if disconnected)
    rationale: str           # human/LLM-readable ordering reason


@dataclass
class ContextResult:
    items: list[ContextItem]
    # DESCRIPTIVE score spread (std of candidate scores). NOTE: measured NOT
    # to distinguish a correct from a wrong anchor -- do not gate on it.
    discrimination: float
    n_candidates: int
    anchors: list[int]
    mode: str


class GraphHandle:
    """An embedded neighborhood: node embeddings + lazy adjacency/BFS."""

    def __init__(self, emb: torch.Tensor, edge_index: np.ndarray,
                 node_ids: np.ndarray, c):
        self.emb = emb
        self.c = c
        self.n = emb.shape[0]
        self.node_ids = node_ids
        self._adj: list[list[int]] = [[] for _ in range(self.n)]
        for s, t in zip(edge_index[0], edge_index[1]):
            self._adj[int(s)].append(int(t))
            self._adj[int(t)].append(int(s))
        self._dcache: dict[int, np.ndarray] = {}

    def hops(self, src: int) -> np.ndarray:
        if src not in self._dcache:
            d = np.full(self.n, -1, dtype=np.int64)
            d[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self._adj[u]:
                    if d[v] < 0:
                        d[v] = d[u] + 1
                        q.append(v)
            self._dcache[src] = d
        return self._dcache[src]

    def min_hops(self, anchors: list[int]) -> np.ndarray:
        hs = [self.hops(a) for a in anchors]
        out = np.full(self.n, -1, dtype=np.int64)
        for h in hs:
            reach = h >= 0
            cur = out < 0
            out[reach & cur] = h[reach & cur]
            both = reach & ~cur
            out[both] = np.minimum(out[both], h[both])
        return out


class KGRContextService:
    def __init__(self, encoder_dir: str = FROZEN, device: str = "cpu"):
        self.encoder_dir = Path(encoder_dir)
        self.device = torch.device(device)
        self._enc = None  # built lazily against the first graph's dims

    def load_graph(self, npz_path_or_dict) -> GraphHandle:
        """Embed a query neighborhood. Accepts an npz path or a dict of
        the tier1 npz arrays (x, edge_index, edge_type, descriptors, ...).
        Uses the FROZEN encoder; never trains."""
        if isinstance(npz_path_or_dict, (str, Path)):
            z = dict(np.load(npz_path_or_dict, allow_pickle=True))
        else:
            z = dict(npz_path_or_dict)
        g = _build_graph_tensors(z)
        if self._enc is None:
            self._enc, _ = _build_encoder(self.encoder_dir, g, self.device)
        with torch.no_grad():
            emb = self._enc(
                g["x"].to(self.device), g["edge_index"].to(self.device),
                g["edge_type"].to(self.device),
                g["edge_descriptor"].to(self.device),
                node_descriptor=g["node_descriptor"].to(self.device),
            ).node_embeddings
        n = emb.shape[0]
        node_ids = (z["neo4j_node_id"] if "neo4j_node_id" in z
                    else np.arange(n))
        return GraphHandle(emb, g["edge_index"].cpu().numpy(),
                           np.asarray(node_ids), self._enc.c)

    def order_context(self, handle: GraphHandle, anchor_rows,
                      top_k: int = 20, ball_hops: int | None = None,
                      exclude_anchors: bool = True) -> ContextResult:
        """Rank context nodes by structural relevance to the anchor(s).

        anchor_rows: one row or a list (multi-anchor -> min-distance).
        ball_hops: if set, restrict candidates to nodes within this many
            hops of any anchor (the Cypher/BFS ball); else score all nodes.
        Returns a deterministic top-k with per-item rationale.
        """
        anchors = ([anchor_rows] if isinstance(anchor_rows, int)
                   else list(anchor_rows))
        if not anchors:
            raise ValueError("at least one anchor row is required")
        emb, c = handle.emb, handle.c

        # multi-anchor score = max over anchors of -dist (== min distance)
        per_anchor = torch.stack(
            [score_from_embeddings(emb, emb[a], c=c) for a in anchors], dim=0)
        score = per_anchor.max(0).values                      # (n,)
        mh = handle.min_hops(anchors)

        cand = np.arange(handle.n)
        if ball_hops is not None:
            cand = cand[(mh >= 0) & (mh <= ball_hops)]
        if exclude_anchors:
            aset = set(anchors)
            cand = np.array([r for r in cand if r not in aset], dtype=np.int64)
        if cand.size == 0:
            return ContextResult([], 0.0, 0, anchors, "order")

        sc = score[cand]
        # deterministic order: sort by (-score, canonical node id).
        ids = handle.node_ids[cand]
        id_key = np.asarray([str(x) for x in ids])
        order = sorted(range(len(cand)),
                       key=lambda i: (-float(sc[i]), id_key[i]))
        order = order[:top_k]

        disc = float(sc.std().item()) if sc.numel() > 1 else 0.0
        items = []
        for rank, i in enumerate(order, 1):
            r = int(cand[i])
            hop = int(mh[r]) if mh[r] >= 0 else None
            items.append(ContextItem(
                node_id=handle.node_ids[r], row=r, rank=rank,
                score=float(sc[i]), hop=hop,
                rationale=_rationale(hop, len(anchors))))
        return ContextResult(items, disc, int(cand.size), anchors, "order")

    def suggest_missing_links(self, handle: GraphHandle, anchor_row: int,
                              top_k: int = 20, min_hop: int = 2
                              ) -> ContextResult:
        """Code-graph capability: rank NON-adjacent nodes (hop>=min_hop)
        by structural affinity to the anchor -- surfacing connections that
        SHOULD exist but don't (edge-missing regime: emb beats hop 10x)."""
        emb, c = handle.emb, handle.c
        h = handle.hops(anchor_row)
        score = score_from_embeddings(emb, emb[anchor_row], c=c)
        cand = np.array([r for r in range(handle.n)
                         if r != anchor_row and (h[r] < 0 or h[r] >= min_hop)],
                        dtype=np.int64)
        if cand.size == 0:
            return ContextResult([], 0.0, 0, [anchor_row], "missing_link")
        sc = score[cand]
        ids = handle.node_ids[cand]
        id_key = np.asarray([str(x) for x in ids])
        order = sorted(range(len(cand)),
                       key=lambda i: (-float(sc[i]), id_key[i]))[:top_k]
        disc = float(sc.std().item()) if sc.numel() > 1 else 0.0
        items = []
        for rank, i in enumerate(order, 1):
            r = int(cand[i])
            hop = int(h[r]) if h[r] >= 0 else None
            items.append(ContextItem(
                node_id=handle.node_ids[r], row=r, rank=rank,
                score=float(sc[i]), hop=hop,
                rationale=f"structurally near anchor but {hop if hop else 'un'}"
                          f"-hop away -> candidate missing link"))
        return ContextResult(items, disc, int(cand.size),
                             [anchor_row], "missing_link")


def _rationale(hop: int | None, n_anchors: int) -> str:
    base = ("adjacent to anchor" if hop == 1 else
            f"{hop} hops from nearest anchor" if hop else
            "structurally near anchor (disconnected in hops)")
    if n_anchors > 1:
        base += f" (min over {n_anchors} anchors)"
    return base
