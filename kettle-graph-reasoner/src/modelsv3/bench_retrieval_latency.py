r"""Experiment 2.4 — retrieval latency benchmark.

Measures per-query latency for two architectures:

    v2 (score-first)      : full forward pass per query producing per-node
                             scores directly.
    v3 (embedding-first)  : encoder runs once per graph (amortized);
                             per-query cost is QueryToBall + distance + topk.

Reports under two framings:

    Choice A — production timing
        v2 is timed as-deployed: full forward per query. v3 is timed with
        encoder pre-computation done once (untimed, reported separately as
        setup cost). Single-query and batch-10 conditions.

    Choice B — query-dependent portion only
        v2's forward is split into (a) a query-independent portion that
        can be cached per graph (node_in, schema_encoder, attn/mp layers),
        and (b) a query-dependent portion (query_in, dist, sigmoid, edge
        aggregation). Choice B times only (b), matching v3's per-query
        path. This closes the "couldn't you just refactor v2?" loophole.

Timing methodology:
    - CPU only (per deployment context).
    - 100 warm-up iterations, then 1000 timed iterations.
    - time.perf_counter_ns() for sub-microsecond resolution.
    - Report median / p50 / p99 / mean / std.
    - Benchmark target: the median-size val graph, so numbers are
      representative rather than optimistic or pessimistic.

Usage
-----
    py src/modelsv3/bench_retrieval_latency.py \\
        --v2-checkpoint runs/compare_task2/hyp_seed_0/best.pt \\
        --v3-checkpoint runs/v3_hyp_compute_seed0/encoder.pt \\
        --v3-summary    runs/v3_hyp_compute_seed0/summary.json \\
        --out           runs/bench_latency.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.models.hyperbolic_gnn_clean import KettleGraphReasonerClean  # noqa: E402
from src.modelsv2.layers import poincare_ops as P  # noqa: E402
from src.modelsv3.distance_scoring import score_from_embeddings  # noqa: E402
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3  # noqa: E402
from src.modelsv3.query_encoder import QueryToBall  # noqa: E402


WARMUP = 100
ITERATIONS = 1000


@dataclass
class TimingResult:
    median_ns: float
    p50_ns: float
    p99_ns: float
    mean_ns: float
    std_ns: float
    min_ns: float
    max_ns: float
    n: int

    def as_dict(self) -> dict:
        return {
            "median_us": self.median_ns / 1000.0,
            "p50_us": self.p50_ns / 1000.0,
            "p99_us": self.p99_ns / 1000.0,
            "mean_us": self.mean_ns / 1000.0,
            "std_us": self.std_ns / 1000.0,
            "min_us": self.min_ns / 1000.0,
            "max_us": self.max_ns / 1000.0,
            "n": self.n,
        }


def _time_callable(fn: Callable[[], None], iterations: int = ITERATIONS) -> TimingResult:
    """Time ``fn`` with warm-up. ``fn`` must take no arguments and must
    produce any side effect needed to prevent dead-code elimination."""
    for _ in range(WARMUP):
        fn()
    samples: list[int] = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
    samples.sort()
    n = len(samples)
    return TimingResult(
        median_ns=samples[n // 2],
        p50_ns=samples[n // 2],
        p99_ns=samples[int(0.99 * (n - 1))],
        mean_ns=statistics.mean(samples),
        std_ns=statistics.stdev(samples) if n > 1 else 0.0,
        min_ns=samples[0],
        max_ns=samples[-1],
        n=n,
    )


def _median_size_graph_idx(dataset: CorpusDataset) -> int:
    """Pick the val graph whose node count is closest to the median across
    all val graphs. More representative than picking graph 0."""
    seen: set[int] = set()
    sizes: list[tuple[int, int]] = []
    for graph_idx, _task_idx in dataset.index:
        gi = int(graph_idx)
        if gi in seen:
            continue
        seen.add(gi)
        graph = dataset._get_graph(gi)
        sizes.append((gi, int(graph["x"].size(0))))
    sizes.sort(key=lambda t: t[1])
    return sizes[len(sizes) // 2][0]


def _load_v2(checkpoint: Path, dataset: CorpusDataset) -> KettleGraphReasonerClean:
    """Reconstruct a v2 model and load its checkpoint. Assumes the default
    v2 config (hidden_dim=64, num_layers=3, type_dim=8, c=1.0) which
    matches the 44353-param summary.json."""
    model = KettleGraphReasonerClean(
        node_feat_dim=dataset.node_feat_dim,
        edge_feat_dim=dataset.edge_feat_dim_schema,
        query_dim=dataset.query_dim,
        hidden_dim=64,
        num_layers=3,
        type_dim=8,
        c=1.0,
        num_edge_types_max=dataset.num_edge_types_max,
        node_feat_dim_schema=dataset.node_feat_dim_schema,
    )
    state = torch.load(checkpoint, map_location="cpu")
    # v2's train loop saves {"epoch", "model_state", "cfg", "val"}; bench-time
    # we only want the parameters. Also tolerate a raw state_dict.
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v2] loaded from {checkpoint} ({n_params} params)")
    return model


def _load_v3(
    checkpoint: Path, summary: Path, dataset: CorpusDataset
) -> tuple[KettleGraphReasonerV3, QueryToBall]:
    """Reconstruct the v3 encoder from its summary.json + checkpoint.
    Also build a fresh (untrained) QueryToBall, since the benchmark is
    about inference cost, not accuracy — an untrained query head has the
    same forward cost as a trained one."""
    with open(summary, "r") as f:
        s = json.load(f)
    cfg = s["config"]
    encoder = KettleGraphReasonerV3(
        node_feat_dim=dataset.node_feat_dim,
        edge_feat_dim=dataset.edge_feat_dim_schema,
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        type_dim=int(cfg["type_dim"]),
        c=float(cfg["curvature"]),
        num_edge_types_max=dataset.num_edge_types_max,
        node_feat_dim_schema=dataset.node_feat_dim_schema,
        tangent_scale_init=float(cfg.get("tangent_scale", 0.1)),
    )
    state = torch.load(checkpoint, map_location="cpu")
    encoder.load_state_dict(state)
    encoder.eval()

    query_head = QueryToBall(
        query_dim=dataset.query_dim,
        hidden_dim=int(cfg["hidden_dim"]),
        c=float(cfg["curvature"]),
        euclidean=False,
    )
    query_head.eval()

    n_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_query = sum(p.numel() for p in query_head.parameters() if p.requires_grad)
    print(f"[v3] encoder loaded from {checkpoint} ({n_encoder} params)")
    print(f"[v3] query head fresh ({n_query} params)")
    return encoder, query_head


def _bench_v2_choice_a(
    model: KettleGraphReasonerClean, sample_graph: dict, query: torch.Tensor, batch_size: int
) -> TimingResult:
    """Choice A for v2: full forward pass per query.

    Batch mode: sequential per-query forward calls, summed. (The v2 class
    doesn't natively batch queries — this reflects actual deployment.)"""
    x = sample_graph["x"]
    edge_index = sample_graph["edge_index"]
    edge_type = sample_graph["edge_type"]
    edge_descriptor = sample_graph["edge_descriptor"]
    node_descriptor = sample_graph["node_descriptor"]

    sink: list[torch.Tensor | None] = [None]

    if batch_size == 1:
        def run() -> None:
            with torch.no_grad():
                out = model(
                    x, edge_index, edge_type, edge_descriptor,
                    query, node_descriptor=node_descriptor,
                )
                # topk to match v3's retrieval output
                _, top = torch.topk(out.node_logits, k=10, largest=True)
                sink[0] = top
    else:
        def run() -> None:
            with torch.no_grad():
                for _ in range(batch_size):
                    out = model(
                        x, edge_index, edge_type, edge_descriptor,
                        query, node_descriptor=node_descriptor,
                    )
                    _, top = torch.topk(out.node_logits, k=10, largest=True)
                sink[0] = top
    return _time_callable(run)


def _bench_v3_choice_a(
    encoder: KettleGraphReasonerV3,
    query_head: QueryToBall,
    sample_graph: dict,
    query: torch.Tensor,
    batch_size: int,
) -> tuple[TimingResult, TimingResult]:
    """Choice A for v3:
        setup (timed separately): encoder forward to produce node_embeddings
        per-query (timed here): query_head + distance + topk
    """
    x = sample_graph["x"]
    edge_index = sample_graph["edge_index"]
    edge_type = sample_graph["edge_type"]
    edge_descriptor = sample_graph["edge_descriptor"]
    node_descriptor = sample_graph["node_descriptor"]

    # Pre-compute the encoder state so per-query cost is only the
    # query-dependent path.
    with torch.no_grad():
        out = encoder(
            x, edge_index, edge_type, edge_descriptor,
            node_descriptor=node_descriptor,
        )
        node_emb = out.node_embeddings
        c_val = encoder.c

    def setup() -> None:
        with torch.no_grad():
            _ = encoder(
                x, edge_index, edge_type, edge_descriptor,
                node_descriptor=node_descriptor,
            )

    setup_timing = _time_callable(setup, iterations=50)  # fewer iters; setup is heavy

    sink: list[torch.Tensor | None] = [None]

    if batch_size == 1:
        def run() -> None:
            with torch.no_grad():
                q_point = query_head(query)
                scores = score_from_embeddings(node_emb, q_point, c=c_val, euclidean=False)
                _, top = torch.topk(scores, k=10, largest=True)
                sink[0] = top
    else:
        def run() -> None:
            with torch.no_grad():
                for _ in range(batch_size):
                    q_point = query_head(query)
                    scores = score_from_embeddings(node_emb, q_point, c=c_val, euclidean=False)
                    _, top = torch.topk(scores, k=10, largest=True)
                sink[0] = top

    query_timing = _time_callable(run)
    return setup_timing, query_timing


def _bench_v2_choice_b(
    model: KettleGraphReasonerClean, sample_graph: dict, query: torch.Tensor, batch_size: int
) -> TimingResult:
    """Choice B for v2: refactored to cache the query-independent portion.

    Cached once (not timed): node_in, expmap0 for nodes, schema_encoder,
                              L layers of attn+mp → final h
    Per query (timed):       query_in, expmap0 for query, dist, sigmoid,
                              edge endpoint average, topk
    """
    x = sample_graph["x"]
    edge_index = sample_graph["edge_index"]
    edge_type = sample_graph["edge_type"]
    edge_descriptor = sample_graph["edge_descriptor"]
    node_descriptor = sample_graph["node_descriptor"]

    # Pre-compute the query-independent portion of v2.
    with torch.no_grad():
        c = model.c
        h_tan = model.node_in(x) * model.tangent_scale
        h = P.expmap0(h_tan, c)
        edge_type_emb, _ = model.schema_encoder(edge_descriptor, node_descriptor)
        for attn, mp in zip(model.attn_layers, model.mp_layers):
            alpha = attn(h, edge_index, edge_type, type_emb_override=edge_type_emb)
            h = mp(h, edge_index, edge_weight=alpha)
        # h is now the final encoder state, analogous to v3's node_embeddings.
    src, dst = edge_index[0], edge_index[1]
    tangent_scale = model.tangent_scale

    sink: list[torch.Tensor | None] = [None]

    if batch_size == 1:
        def run() -> None:
            with torch.no_grad():
                q_flat = query.view(-1)
                q_tan = model.query_in(q_flat) * tangent_scale
                h_q = P.expmap0(q_tan, c)
                N = h.size(0)
                h_q_exp = h_q.unsqueeze(0).expand(N, -1)
                d = P.dist(h, h_q_exp, c)
                node_logits = -d
                _, top = torch.topk(node_logits, k=10, largest=True)
                sink[0] = top
    else:
        def run() -> None:
            with torch.no_grad():
                for _ in range(batch_size):
                    q_flat = query.view(-1)
                    q_tan = model.query_in(q_flat) * tangent_scale
                    h_q = P.expmap0(q_tan, c)
                    N = h.size(0)
                    h_q_exp = h_q.unsqueeze(0).expand(N, -1)
                    d = P.dist(h, h_q_exp, c)
                    node_logits = -d
                    _, top = torch.topk(node_logits, k=10, largest=True)
                sink[0] = top
    return _time_callable(run)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2-checkpoint", type=str, required=True)
    p.add_argument("--v3-checkpoint", type=str, required=True)
    p.add_argument("--v3-summary", type=str, required=True)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--task", type=int, default=2)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    torch.set_num_threads(1)  # single-threaded for reproducible timings
    dataset = CorpusDataset(
        corpus_dir=args.corpus,
        split=args.split,
        split_seed=args.split_seed,
        include_tasks={args.task},
    )
    print(f"[bench] dataset: split={args.split} samples={len(dataset)}")

    # Pick a median-size graph.
    target_gi = _median_size_graph_idx(dataset)
    print(f"[bench] target graph: {target_gi}")
    graph = dataset._get_graph(target_gi)
    n_nodes = int(graph["x"].size(0))
    n_edges = int(graph["edge_index"].size(1))
    # Any task sample on this graph has a query we can use.
    query_sample = next(
        dataset[i] for i in range(len(dataset)) if int(dataset.index[i][0]) == target_gi
    )
    query = query_sample.query
    print(f"[bench] graph shape: N={n_nodes} E={n_edges} query_dim={query.size(-1)}")

    # Load models.
    v2 = _load_v2(Path(args.v2_checkpoint), dataset)
    v3_enc, v3_q = _load_v3(Path(args.v3_checkpoint), Path(args.v3_summary), dataset)

    # -- Choice A: production timings --
    print("\n[Choice A] v2 single-query...")
    v2_a_single = _bench_v2_choice_a(v2, graph, query, batch_size=1)
    print("[Choice A] v2 batch-10...")
    v2_a_batch = _bench_v2_choice_a(v2, graph, query, batch_size=10)
    print("[Choice A] v3 setup cost...")
    v3_setup, v3_a_single = _bench_v3_choice_a(v3_enc, v3_q, graph, query, batch_size=1)
    print("[Choice A] v3 batch-10...")
    _, v3_a_batch = _bench_v3_choice_a(v3_enc, v3_q, graph, query, batch_size=10)

    # -- Choice B: query-dependent portion only --
    print("\n[Choice B] v2 (refactored, query-dep only) single-query...")
    v2_b_single = _bench_v2_choice_b(v2, graph, query, batch_size=1)
    print("[Choice B] v2 (refactored, query-dep only) batch-10...")
    v2_b_batch = _bench_v2_choice_b(v2, graph, query, batch_size=10)
    # v3 Choice B is the same as Choice A since v3 is *already* decomposed.

    results = {
        "graph": {"idx": target_gi, "n_nodes": n_nodes, "n_edges": n_edges},
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "choice_a_production": {
            "v2_single": v2_a_single.as_dict(),
            "v2_batch10": v2_a_batch.as_dict(),
            "v3_single": v3_a_single.as_dict(),
            "v3_batch10": v3_a_batch.as_dict(),
            "v3_setup_per_graph": v3_setup.as_dict(),
        },
        "choice_b_query_only": {
            "v2_single": v2_b_single.as_dict(),
            "v2_batch10": v2_b_batch.as_dict(),
            "v3_single": v3_a_single.as_dict(),  # same path as Choice A
            "v3_batch10": v3_a_batch.as_dict(),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    _print_table(results)
    return 0


def _print_table(r: dict) -> None:
    print()
    print("=" * 84)
    print(f"EXPERIMENT 2.4 — Retrieval latency (graph {r['graph']['idx']}, "
          f"N={r['graph']['n_nodes']}, E={r['graph']['n_edges']})")
    print("=" * 84)

    def row(label: str, ts: dict) -> str:
        return (
            f"{label:<28} median={ts['median_us']:8.1f}µs  "
            f"p99={ts['p99_us']:8.1f}µs  mean={ts['mean_us']:8.1f}µs  "
            f"std={ts['std_us']:7.1f}µs"
        )

    print("\nChoice A — production timing (v2 full forward vs v3 pre-computed encoder)")
    print("-" * 84)
    print(row("v2 full forward (single)", r["choice_a_production"]["v2_single"]))
    print(row("v2 full forward (batch 10)", r["choice_a_production"]["v2_batch10"]))
    print(row("v3 post-encode (single)", r["choice_a_production"]["v3_single"]))
    print(row("v3 post-encode (batch 10)", r["choice_a_production"]["v3_batch10"]))
    print(row("v3 encoder setup (per graph)", r["choice_a_production"]["v3_setup_per_graph"]))

    # Speedup on single-query production path.
    v2a = r["choice_a_production"]["v2_single"]["median_us"]
    v3a = r["choice_a_production"]["v3_single"]["median_us"]
    setup = r["choice_a_production"]["v3_setup_per_graph"]["median_us"]
    speedup = v2a / v3a if v3a > 0 else float("nan")
    breakeven = setup / (v2a - v3a) if v2a > v3a else float("nan")
    print(f"\n    v3 speedup (single-query): {speedup:.1f}×")
    print(f"    v3 setup break-even: after {breakeven:.1f} queries on the same graph")

    print("\nChoice B — query-dependent portion only (v2 refactored, cached encoder)")
    print("-" * 84)
    print(row("v2 query-only (single)", r["choice_b_query_only"]["v2_single"]))
    print(row("v2 query-only (batch 10)", r["choice_b_query_only"]["v2_batch10"]))
    print(row("v3 query-only (single)", r["choice_b_query_only"]["v3_single"]))
    print(row("v3 query-only (batch 10)", r["choice_b_query_only"]["v3_batch10"]))

    v2b = r["choice_b_query_only"]["v2_single"]["median_us"]
    v3b = r["choice_b_query_only"]["v3_single"]["median_us"]
    speedup_b = v2b / v3b if v3b > 0 else float("nan")
    print(f"\n    v3 speedup (query-only, single): {speedup_b:.2f}×")
    print("    (if ≈1.0×, Choice A win is entirely from precomputation, not geometry)")


if __name__ == "__main__":
    sys.exit(main())
