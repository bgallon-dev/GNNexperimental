r"""Candidate ordering strategies over a projected bundle (plan T4/T5).

Three lanes exist: ``bfs`` (the shipping default for every family/lane
cell), ``lexical`` and ``kgr``. Only BFS ships until T4's pre-registered
paired-confidence gate says otherwise -- the other lanes are implemented
here so the evaluation harness can run, not because they are adopted.

All scores are quantized to integer micro-scores before sorting and ties
break on canonical public key (Determinism policy). The KGR lane's
embeddings come from the SHA-asserted frozen encoder over the SAME
bounded pull the bundle was projected from -- never an ``end_to_end``
projection."""

from __future__ import annotations

import re
from collections import deque
from typing import Callable, Mapping

from . import canonical
from .contracts import CandidateBundle, RankedCandidate, ResearchQuestion


def _per_anchor_hops(bundle: CandidateBundle) -> dict[str, dict[str, int]]:
    """anchor key -> {node key -> hop} over the bundle's domain edges
    (exact graph distance within the projection; deterministic)."""
    adj: dict[str, set[str]] = {n.public_key: set() for n in bundle.nodes}
    for r in bundle.relationships:
        adj[r.start_key].add(r.end_key)
        adj[r.end_key].add(r.start_key)
    out: dict[str, dict[str, int]] = {}
    for a in bundle.anchors:
        hops = {a: 0}
        q = deque([a])
        while q:
            u = q.popleft()
            for v in sorted(adj[u]):
                if v not in hops:
                    hops[v] = hops[u] + 1
                    q.append(v)
        out[a] = hops
    return out


def bfs_ranking(bundle: CandidateBundle) -> list[RankedCandidate]:
    """Deterministic BFS ordering: (min hop over anchors, public key).

    ``contributing_anchors`` lists every anchor achieving the min hop --
    the compiler's round-robin core selection groups on the first one."""
    per_anchor = _per_anchor_hops(bundle)
    rows = []
    for n in bundle.nodes:
        d = {a: h[n.public_key] for a, h in per_anchor.items()
             if n.public_key in h}
        hop = min(d.values()) if d else n.hop
        nearest = tuple(sorted(a for a, v in d.items() if v == hop))
        rows.append((hop, n.public_key, nearest))
    rows.sort(key=lambda t: (t[0], t[1]))
    return [
        RankedCandidate(
            public_key=key, strategy="bfs", rank=i + 1,
            micro_score=canonical.to_micro_score(1.0 / (1.0 + hop)),
            hop=hop, contributing_anchors=nearest, bfs_rank=i + 1,
            rationale=f"bfs hop {hop} from "
                      f"{nearest[0] if nearest else 'no anchor'}")
        for i, (hop, key, nearest) in enumerate(rows)]


# -- lexical lane ---------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOP = frozenset("the and for was that with this from have has are were "
                  "did does how what which who whom when where why".split())


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower())} - _STOP


def lexical_ranking(bundle: CandidateBundle,
                    question: ResearchQuestion) -> list[RankedCandidate]:
    """Deterministic question-term affinity: overlap between the question's
    content tokens and each domain node's own text PLUS the text of its
    attached claim-layer neighbors. Scores in [0,1] (matched fraction of
    question tokens), micro-quantized, key tie-breaks."""
    q_tokens = _tokens(question.text)
    prov_text: dict[str, str] = {
        n.public_key: " ".join(str(v) for v in n.properties.values())
        for n in bundle.provenance_nodes}
    attached: dict[str, list[str]] = {}
    for r in bundle.provenance_relationships:
        for a, b in ((r.start_key, r.end_key), (r.end_key, r.start_key)):
            if b in prov_text:
                attached.setdefault(a, []).append(prov_text[b])
    hops = {n.public_key: n.hop for n in bundle.nodes}
    rows = []
    for n in bundle.nodes:
        blob = " ".join((
            " ".join(str(v) for v in n.properties.values()),
            *sorted(attached.get(n.public_key, ()))))
        overlap = len(q_tokens & _tokens(blob)) / max(1, len(q_tokens))
        rows.append((canonical.to_micro_score(overlap), n.public_key))
    rows.sort(key=lambda t: (-t[0], t[1]))
    return [RankedCandidate(
        public_key=key, strategy="lexical", rank=i + 1, micro_score=m,
        hop=hops[key], lexical_rank=i + 1,
        rationale=f"lexical overlap {canonical.format_micro_score(m)}")
        for i, (m, key) in enumerate(rows)]


# -- kgr lane ---------------------------------------------------------------------

# embedder: neo4j ids -> {id: embedding row (torch tensor)}; injected so
# the lane is hermetically testable and the heavy frozen-encoder load is
# opt-in (see KGREmbedder below)
Embedder = Callable[[list[int]], Mapping[int, "object"]]


def kgr_ranking(bundle: CandidateBundle,
                embedder: Embedder) -> list[RankedCandidate]:
    """Frozen-encoder ordering of the projected ball: max-union hyperbolic
    affinity to ANY confirmed anchor (the shipped ``multi_anchor_order``
    composition, +0.471 verified; per-endpoint min/relay compositions are
    refuted and deliberately absent)."""
    import torch

    from src.modelsv3.retrieval_ops import score_from_embeddings

    ids = [n.neo4j_id for n in bundle.nodes]
    emb = embedder(ids)
    anchor_ids = {n.public_key: n.neo4j_id for n in bundle.nodes
                  if n.public_key in bundle.anchors}
    hops = {n.public_key: n.hop for n in bundle.nodes}
    key_by_id = {n.neo4j_id: n.public_key for n in bundle.nodes}
    scored_ids = [i for i in ids if i in emb]
    node_emb = torch.stack([torch.as_tensor(emb[i]) for i in scored_ids])
    scores = torch.stack([
        score_from_embeddings(node_emb, torch.as_tensor(emb[a]))
        for a in anchor_ids.values() if a in emb]).max(dim=0).values
    rows = sorted(
        ((canonical.to_micro_score(float(s)), key_by_id[i])
         for i, s in zip(scored_ids, scores)),
        key=lambda t: (-t[0], t[1]))
    return [RankedCandidate(
        public_key=key, strategy="kgr", rank=r + 1, micro_score=m,
        hop=hops[key], contributing_anchors=tuple(sorted(anchor_ids)),
        rationale="frozen-encoder max-union anchor affinity")
        for r, (m, key) in enumerate(rows)]


class KGREmbedder:
    """Live embedder: re-encodes the bundle's EXACT node set with the
    SHA-asserted frozen baseline encoder via the parity-proven path
    (pull_by_ids -> encode_subgraph -> _build_graph_tensors -> encoder;
    bit-exact to the reference pipeline per verify.py P1)."""

    def __init__(self, evidence_source):
        import json as _json
        from pathlib import Path
        from types import SimpleNamespace

        import torch

        from src.modelsv3.eval_candidate_recall import _build_encoder
        from src.modelsv3.lock_baseline import assert_encoder_sha

        root = Path(__file__).resolve().parents[2]
        baseline = root / "runs" / "v3.1-baseline-hyp-h128-l4-seed1"
        cfg = _json.loads((baseline / "summary.json").read_text())["config"]
        ns = SimpleNamespace(node_feat_dim=32, edge_feat_dim_schema=13,
                             node_feat_dim_schema=4, num_edge_types_max=30,
                             query_dim=18)
        assert_encoder_sha(baseline, baseline / "encoder.pt")
        self._enc = _build_encoder(cfg, ns)
        self._enc.load_state_dict(
            torch.load(baseline / "encoder.pt", map_location="cpu"))
        self._enc.eval()
        self._src = evidence_source._src        # tensor-parity Neo4jSource

    def __call__(self, ids: list[int]):
        import torch

        from src.data.corpus_dataset import _build_graph_tensors
        from src.service.schema_map import SchemaMap
        from src.service.tensor_contract import encode_subgraph

        pull = self._src.pull_by_ids(list(ids))
        npz_like = encode_subgraph(pull, SchemaMap.from_yaml())
        g = _build_graph_tensors(npz_like)
        with torch.no_grad():
            emb = self._enc(g["x"], g["edge_index"], g["edge_type"],
                            g["edge_descriptor"],
                            node_descriptor=g["node_descriptor"]
                            ).node_embeddings
        return {int(i): emb[r] for r, i in
                enumerate(npz_like["neo4j_node_id"])}


# -- lanes -----------------------------------------------------------------------

def lane_rankings(ranking: list[RankedCandidate]) -> dict[str, list[str]]:
    """The two observable serving lanes (plan T4): the full ranking, and
    the hop>=2 subranking. Hop comes from the bundle -- observable at
    serving time; ground-truth locality is never used."""
    return {
        "core": [rc.public_key for rc in ranking],
        "nonlocal_discovery": [rc.public_key for rc in ranking
                               if rc.hop >= 2],
    }
