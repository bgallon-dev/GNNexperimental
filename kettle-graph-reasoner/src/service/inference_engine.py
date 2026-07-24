r"""KGRRetriever -- the in-process serving engine.

Wires the SHA-asserted frozen v3.1 encoder + the per-task QueryToBall
head + the gated reranker recipe into one ``retrieve()`` call:

    live Neo4j pull -> tensor_contract -> frozen encoder forward
    -> single-graph ManifoldIndex -> query_point -> route
       (tasks 0/5: hybrid_retrieve_expand_rerank; tasks 2/4:
        retrieve_then_rerank) -> ranked Neo4j ids + scores + metadata

No model code is touched: the encoder forward call is bit-identical to
``export_manifold_index.py:135`` and the retrieval ops are reused
unchanged. The encoder SHA is asserted against the locked baseline
manifest at construction (``lock_baseline.assert_encoder_sha``) so the
served weights are provably the ``ed8139dc8209...`` frozen baseline that
every head was trained against.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from . import IN_SCOPE_TASKS, OUT_OF_SCOPE_TASKS
from .routing import Routing
from .schema_map import SchemaMap
from .tensor_contract import build_graph_tensors, encode_query, encode_subgraph

# Contract dims (corpus_dataset.py authoritative; kept local -> corpus-free).
_CONTRACT = SimpleNamespace(
    node_feat_dim=32, edge_feat_dim_schema=13, node_feat_dim_schema=4,
    num_edge_types_max=30, query_dim=18,
)
# tasks that benefit from provenance expansion before ranking
_EXPAND_TASKS = {0, 5}


@dataclass
class RetrievalResult:
    task: int
    ranked: list[tuple[int, float]]          # (neo4j_id, score), best first
    recipe: str
    head_run: str
    encoder_sha: str
    n_subgraph_nodes: int
    n_subgraph_edges: int
    expected_ndcg: Optional[float]
    routing_reason: str
    latencies_ms: dict = field(default_factory=dict)


class KGRRetriever:
    def __init__(
        self,
        routing_path: str | Path | None = None,
        schema_map_path: str | Path | None = None,
        subgraph: str = "domain_only",
        config_path: str | Path | None = None,
    ) -> None:
        import torch

        from .determinism import hashseed_ok

        if not hashseed_ok():
            print("[KGRRetriever] WARNING: PYTHONHASHSEED != 0. Edge-type "
                  "slot order (sorted(set(rel_types))) is hash-randomized "
                  "for tied counts, so live encodes are not reproducible "
                  "and may not match the reference. Run via "
                  "`py -m src.service.kgr_retrieve` (auto-pins) or set "
                  "PYTHONHASHSEED=0 before launching Python.")

        self.routing = Routing.from_yaml(routing_path)
        self.schema_map = SchemaMap.from_yaml(schema_map_path)
        self.subgraph = subgraph
        self._config_path = config_path

        baseline_dir = self.routing.baseline_dir
        enc_pt = baseline_dir / "encoder.pt"
        summary = baseline_dir / "summary.json"
        for p in (enc_pt, summary,
                  baseline_dir / "baseline_manifest.json"):
            if not p.exists():
                raise FileNotFoundError(f"baseline asset missing: {p}")

        # SHA-assert: the served encoder IS the locked frozen baseline.
        from src.modelsv3.lock_baseline import (  # noqa: E402
            assert_encoder_sha,
            sha256_file,
        )
        assert_encoder_sha(baseline_dir, enc_pt)
        self.encoder_sha = sha256_file(enc_pt)

        import json

        cfg = json.loads(summary.read_text())["config"]
        self._cfg = cfg
        self.euclidean = cfg["model"] == "euclidean"

        from src.modelsv3.eval_candidate_recall import _build_encoder  # noqa: E402

        self.encoder = _build_encoder(cfg, _CONTRACT)
        self.encoder.load_state_dict(
            torch.load(enc_pt, map_location="cpu"))
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.c_val = getattr(
            self.encoder, "c",
            torch.tensor(float(cfg.get("curvature", 1.0))))

        # selectable per-task heads (load_query_encoder) -- cached.
        from src.modelsv3.retrieval_ops import load_query_encoder  # noqa: E402

        self._load_query_encoder = load_query_encoder
        self._head_cache: dict[int, object] = {}

        self._source = None  # lazy Neo4jSource (the one-time 327k pull)

    # -- lazy live source ---------------------------------------------------

    def _src(self):
        if self._source is None:
            from .neo4j_source import Neo4jSource

            self._source = Neo4jSource(
                config_path=self._config_path,
                schema_map=self.schema_map,
                subgraph=self.subgraph,
            )
        return self._source

    def _head(self, task: int):
        if task not in self._head_cache:
            hd = self.routing.head_dir(task)
            self._head_cache[task] = self._load_query_encoder(hd)
        return self._head_cache[task]

    def close(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    # -- the product call ---------------------------------------------------

    def retrieve(
        self,
        *,
        task: int,
        seed_ids: list[int] | None = None,
        cypher: str | None = None,
        temporal_window: tuple[float, float] | None = None,
        max_hops: int = 4,
        component_tasks: tuple[int, ...] = (),
        k_hops: int = 2,
        max_nodes: int = 400,
        top_k: int = 10,
        candidate_c: int = 50,
    ) -> RetrievalResult:
        import torch

        if task in OUT_OF_SCOPE_TASKS:
            raise ValueError(
                f"task {task} is out of scope: {OUT_OF_SCOPE_TASKS[task]}")
        if task not in IN_SCOPE_TASKS:
            raise ValueError(
                f"task {task} unknown; in scope: {IN_SCOPE_TASKS}")

        t = {}
        t0 = time.perf_counter()

        pull = self._src().pull_subgraph(
            seed_ids=seed_ids, cypher=cypher,
            k_hops=k_hops, max_nodes=max_nodes)
        t["pull"] = (time.perf_counter() - t0) * 1e3

        t1 = time.perf_counter()
        npz_like = encode_subgraph(pull, self.schema_map)
        g = build_graph_tensors(npz_like)
        t["encode_contract"] = (time.perf_counter() - t1) * 1e3

        t2 = time.perf_counter()
        with torch.no_grad():
            emb = self.encoder(
                g["x"], g["edge_index"], g["edge_type"],
                g["edge_descriptor"],
                node_descriptor=g["node_descriptor"],
            ).node_embeddings.detach()
        t["encoder_forward"] = (time.perf_counter() - t2) * 1e3

        # single-graph in-memory ManifoldIndex (reuse retrieval_ops as-is)
        from src.modelsv3.distance_scoring import score_from_embeddings  # noqa: E402
        from src.modelsv3.retrieval_ops import (  # noqa: E402
            ManifoldIndex,
            hybrid_retrieve_expand_rerank,
            identity_reranker,
            retrieve_then_rerank,
        )

        N = emb.size(0)
        node_ids = np.asarray(npz_like["neo4j_node_id"], dtype=np.int64)
        index = ManifoldIndex(
            embedding=emb.cpu().numpy().astype(np.float32),
            graph_idx=np.zeros(N, dtype=np.int64),
            node_idx=np.arange(N, dtype=np.int64),
            neo4j_node_id=node_ids,
            radius=np.zeros(N, dtype=np.float32),
            out_degree=np.bincount(
                g["edge_index"][0].cpu().numpy(), minlength=N),
            in_degree=np.bincount(
                g["edge_index"][1].cpu().numpy(), minlength=N),
            collapse_flag=np.zeros(N, dtype=bool),
            node_type=np.full(N, -1, dtype=np.int64),
            layer=np.full(N, -1, dtype=np.int64),
            depth=np.zeros(N, dtype=np.int64),
            meta={"model": self._cfg["model"],
                  "curvature": float(self.c_val)},
        )

        t3 = time.perf_counter()
        query_np = encode_query(
            task_type=task, temporal_window=temporal_window,
            max_hops=max_hops, component_tasks=component_tasks)
        query_t = torch.from_numpy(query_np)
        query_point = self._head(task)(query_t)
        rs_full = score_from_embeddings(
            emb, query_point, c=self.c_val, euclidean=self.euclidean)
        t["query_score"] = (time.perf_counter() - t3) * 1e3

        decision = self.routing.reranker_decision(task)
        builder = self.routing.make_reranker(decision)

        t4 = time.perf_counter()
        if builder is None:
            reranker = identity_reranker(index, query_point, 0)
        else:
            reranker = builder(
                g, query_t, task, rs_full,
                float(self.c_val), self.euclidean)

        if task in _EXPAND_TASKS:
            rows = hybrid_retrieve_expand_rerank(
                index, query_point, reranker, 0,
                g["edge_index"].cpu().numpy(),
                g["edge_type"].cpu().numpy(),
                g["edge_descriptor"].cpu().numpy(),
                C=candidate_c, k=top_k, expand_hops=1,
            )
        else:
            rows = retrieve_then_rerank(
                index, query_point, reranker, 0,
                C=candidate_c, k=top_k)
        t["rerank"] = (time.perf_counter() - t4) * 1e3
        t["total"] = (time.perf_counter() - t0) * 1e3

        # Score each returned row with the SAME scorer that produced the
        # ordering (the reranker callable is deterministic/stateless), so
        # "best first" results carry monotonically decreasing scores. With
        # the identity reranker this equals rs_full; with a trained blend
        # it is the combined score — never the raw retrieval score, which
        # would be non-monotonic under the reranked order and mislead any
        # consumer explaining the ranking.
        if rows:
            order_scores = np.asarray(
                reranker(np.asarray(rows, dtype=np.int64)), dtype=np.float64)
            ranked = [(int(node_ids[r]), float(s))
                      for r, s in zip(rows, order_scores)]
        else:
            ranked = []
        return RetrievalResult(
            task=task,
            ranked=ranked,
            recipe=decision.recipe,
            head_run=str(self.routing.head_dir(task)),
            encoder_sha=self.encoder_sha,
            n_subgraph_nodes=N,
            n_subgraph_edges=int(g["edge_index"].size(1)),
            expected_ndcg=decision.expected_ndcg,
            routing_reason=decision.reason,
            latencies_ms={k: round(v, 2) for k, v in t.items()},
        )
