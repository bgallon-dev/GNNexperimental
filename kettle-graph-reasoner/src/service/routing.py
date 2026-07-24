r"""Per-task head + (gated) reranker recipe routing.

Two data-driven decisions, each with an explicit no-regression gate:

  head      retrieval_ops.load_query_encoder(head_dir[task]) -- the
            selectable QueryToBall mapping query -> ball point.
  reranker  a trained recipe is deployed for a task ONLY if its real
            router cell is validated (routed >= retriever AND
            regression==false in runs/reranker_router_real/
            router_results.json) AND the artifact's own val gate
            (hybrid.json `residual_deployed`) passed. Any failure ->
            identity reranker (== v3.1-alone), which is provably
            non-regressing vs the retriever (reranker_router.py:102-106
            invariant). The synthetic recipe/task map does NOT transfer
            (PROJECT_HANDOFF sec.2), so this is read from the REAL table,
            never assumed.

Reuses ``reranker_v32._build_v2`` / ``_combine`` and
``retrieval_ops.identity_reranker`` unchanged -- this module is wiring,
not new model/recipe code.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_YAML = Path(__file__).with_name("routing.yaml")

# Contract dims (corpus_dataset.py is authoritative; kept local to stay
# corpus-free at serve time -- a SimpleNamespace is all _build_v2 needs).
_CONTRACT = SimpleNamespace(
    node_feat_dim=32,
    edge_feat_dim_schema=13,     # EDGE_DESC_DIM
    node_feat_dim_schema=4,      # NODE_DESC_DIM
    num_edge_types_max=30,       # MAX_EDGE_TYPES
    query_dim=18,
)


@dataclass
class RerankerDecision:
    task: int
    kind: str                       # "blend" | "identity"
    recipe: str                     # human label for metadata
    artifact: Optional[str]         # reranker.pt dir, if any
    a: Optional[float]
    b: Optional[float]
    v2_cfg: Optional[dict]
    expected_ndcg: Optional[float]  # routed_deployed_mean (real), if validated
    retriever_ndcg: Optional[float]
    regression: bool
    reason: str                     # why this decision (honest metadata)


class Routing:
    def __init__(self, data: dict, source: str | Path | None = None):
        self.source = str(source) if source else "<dict>"
        self.baseline_dir = _ROOT / data["baseline_dir"]
        self.router_results = _ROOT / data["router_results"]
        self._heads = {int(k): v for k, v in (data.get("heads") or {}).items()}
        self._heads_sha = {
            int(k): str(v)
            for k, v in (data.get("heads_sha256") or {}).items()
        }
        self._rerankers = {
            int(k): v for k, v in (data.get("rerankers") or {}).items()
        }
        self._router = None
        self._v2_cache: dict[str, object] = {}

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Routing":
        p = Path(path) if path else _DEFAULT_YAML
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data, source=p)

    # -- head ---------------------------------------------------------------

    def head_dir(self, task: int) -> Path:
        if task not in self._heads:
            raise KeyError(f"no head configured for task {task}")
        d = _ROOT / self._heads[task]
        # optional integrity pin (routing.yaml `heads_sha256:`). Unlike the
        # reranker (which has a safe identity fallback), a wrong HEAD has
        # no non-regressing substitute -- so a pin mismatch is fatal.
        exp = self._heads_sha.get(task)
        if exp:
            from src.modelsv3.lock_baseline import sha256_file  # noqa: E402
            got = sha256_file(d / "query_encoder.pt")
            if got != exp:
                raise RuntimeError(
                    f"head sha mismatch for task {task} "
                    f"({d / 'query_encoder.pt'}): got {got[:12]}.., "
                    f"pinned {exp[:12]}..")
        return d

    # -- reranker decision (the gate) ---------------------------------------

    def _router_table(self) -> dict:
        if self._router is None:
            if not self.router_results.exists():
                self._router = {}
            else:
                payload = json.loads(self.router_results.read_text())
                rep = payload.get("router", payload)
                self._router = (rep.get("by_task") or {})
        return self._router

    def reranker_decision(self, task: int) -> RerankerDecision:
        bt = self._router_table().get(str(task))
        cfg = self._rerankers.get(task)

        def _identity(reason: str,
                      exp: float | None = None,
                      retr: float | None = None) -> RerankerDecision:
            return RerankerDecision(
                task=task, kind="identity",
                recipe="retriever (identity fallback)",
                artifact=None, a=None, b=None, v2_cfg=None,
                expected_ndcg=exp, retriever_ndcg=retr,
                regression=False, reason=reason)

        if bt is None:
            return _identity(
                f"task {task} has no real router cell -> provably-"
                f"non-regressing identity fallback (G2 not yet run)")
        retr = float(bt.get("retriever_ndcg@10_mean")) \
            if bt.get("retriever_ndcg@10_mean") is not None else None
        routed = float(bt.get("routed_deployed_mean")) \
            if bt.get("routed_deployed_mean") is not None else None
        if bool(bt.get("regression")):
            return _identity(
                f"router marks task {task} as regression -> identity",
                routed, retr)
        if routed is not None and retr is not None and routed < retr - 1e-9:
            return _identity(
                f"routed {routed:.4f} < retriever {retr:.4f} -> identity",
                routed, retr)
        if cfg is None:
            return _identity(
                f"task {task} validated in router but no reranker artifact "
                f"wired -> identity (configure routing.yaml rerankers)",
                routed, retr)

        art = _ROOT / cfg["artifact"]
        hybrid = art / "hybrid.json"
        ckpt = art / "reranker.pt"
        if not ckpt.exists():
            return _identity(
                f"reranker artifact missing ({ckpt}) -> identity",
                routed, retr)
        # FAIL CLOSED: the artifact's own val-gate attestation is REQUIRED.
        # A reranker.pt without its hybrid.json is an unattested weight
        # blob; deploying it would bypass the validation gate entirely.
        if not hybrid.exists():
            return _identity(
                f"artifact attestation missing ({hybrid}) -> identity "
                f"(gate metadata is required; never deploy unattested "
                f"weights)", routed, retr)
        hj = json.loads(hybrid.read_text())
        # explicit artifact-identity checks: wrong-task / wrong-recipe
        # artifacts must not deploy even if their own gate passed.
        if "task" in hj and int(hj["task"]) != int(task):
            return _identity(
                f"artifact identity mismatch: hybrid.json task="
                f"{hj['task']} != routed task {task} -> identity",
                routed, retr)
        if hj.get("combine_mode") is not None \
                and str(hj["combine_mode"]) != "blend":
            return _identity(
                f"artifact identity mismatch: combine_mode="
                f"{hj.get('combine_mode')!r} (expected 'blend') -> identity",
                routed, retr)
        if not bool(hj.get("residual_deployed", False)):
            return _identity(
                f"artifact val gate failed (residual_deployed=false) "
                f"-> identity", routed, retr)
        if bool(hj.get("regression_vs_retriever", False)):
            return _identity(
                "artifact regression_vs_retriever=true -> identity",
                routed, retr)
        # optional integrity pin (routing.yaml `sha256:`): a drifted or
        # swapped reranker.pt falls back to identity, loudly.
        exp_sha = cfg.get("sha256")
        if exp_sha:
            from src.modelsv3.lock_baseline import sha256_file  # noqa: E402
            got = sha256_file(ckpt)
            if got != str(exp_sha):
                return _identity(
                    f"artifact sha mismatch ({ckpt}): got {got[:12]}.., "
                    f"pinned {str(exp_sha)[:12]}.. -> identity",
                    routed, retr)

        import torch
        blob = torch.load(ckpt, map_location="cpu")
        v2cfg = blob.get("cfg", {})
        a = blob.get("a")
        b = blob.get("b")
        if a is None or b is None or v2cfg.get("combine_mode") != "blend":
            return _identity(
                f"artifact is not a usable blend reranker -> identity",
                routed, retr)
        return RerankerDecision(
            task=task, kind="blend",
            recipe=str(cfg.get("recipe", "v3.3-blend")),
            artifact=str(art), a=float(a), b=float(b), v2_cfg=dict(v2cfg),
            expected_ndcg=routed, retriever_ndcg=retr,
            regression=False,
            reason=(f"validated real router cell (routed {routed:.4f} >= "
                    f"retriever {retr:.4f}, regression=false); artifact "
                    f"val gate passed"))

    # -- reranker builder ---------------------------------------------------

    def make_reranker(self, decision: RerankerDecision):
        """Return a builder ``fn(graph_t, query_t, task_type, rs_full,
        c, euclidean) -> retrieval_ops.Reranker``.

        For ``identity`` the engine uses ``retrieval_ops.identity_reranker``
        directly (this returns None). For ``blend`` we load the trained v2
        once (process-cached) and return a closure that reproduces
        ``reranker_v32`` eval semantics exactly: combined = _combine(rs,
        cand, v2_scores, (a,b), 'blend'); retrieve_then_rerank then
        argsorts combined[cand] descending -- identical ordering to
        reranker_v32's `_rerank_vec` ndcg path.
        """
        if decision.kind == "identity":
            return None

        from src.modelsv3.reranker_v32 import _build_v2, _combine  # noqa: E402
        import torch

        art = decision.artifact
        if art not in self._v2_cache:
            vc = decision.v2_cfg or {}
            v2 = _build_v2(_CONTRACT, int(vc["hidden_dim"]),
                           int(vc["num_layers"]), int(vc["type_dim"]))
            blob = torch.load(Path(art) / "reranker.pt", map_location="cpu")
            v2.load_state_dict(blob["model_state"])
            v2.eval()
            for p in v2.parameters():
                p.requires_grad = False
            self._v2_cache[art] = v2
        v2 = self._v2_cache[art]
        a = torch.tensor(float(decision.a))
        b = torch.tensor(float(decision.b))

        def _builder(graph_t, query_t, task_type, rs_full, c, euclidean):
            with torch.no_grad():
                v2s = v2(
                    graph_t["x"], graph_t["edge_index"], graph_t["edge_type"],
                    graph_t["edge_descriptor"], query_t,
                    node_descriptor=graph_t["node_descriptor"],
                    task_type=int(task_type),
                ).node_scores.detach()

            def _rr(cand_rows: np.ndarray) -> np.ndarray:
                cand = torch.as_tensor(np.asarray(cand_rows),
                                       dtype=torch.long)
                combined = _combine(rs_full, cand, v2s, (a, b), "blend")
                return combined[cand].detach().cpu().numpy()

            return _rr

        return _builder
