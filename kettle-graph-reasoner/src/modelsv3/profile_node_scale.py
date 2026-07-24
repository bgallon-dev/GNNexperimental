r"""v3.1 — neighborhood-SIZE scaling profiler.

The second scale axis (after graph-count). Holds the v3.1 model fixed
(frozen baseline encoder + qh1 head) and grows the per-graph node cap:

    400 -> 800 -> 1500 -> 3000 -> 5000   (hardened/delocalized sampler)

At each rung it measures BOTH resource cost and quality:

  encode_time_s      mean encoder forward wall-time / graph
  peak_rss_mb        peak process RSS during encode+metrics (psutil
                     sampler thread) and the analytic size of the
                     would-be full NxN distance matrix (why we chunk)
  ndcg@10 / @20, recall@50 / @100, edge_prec@5, bridge_hit@5,
  collapse_rate, spearman(emb_dist, graph_hop), hub degree pattern
  (edge_prec@5 high vs low out-degree tercile)

Why this file exists (the bottleneck the user flagged): the shipped
eval scripts build a full ``(N, N)`` Poincare distance matrix. The
encoder is O(edges) (these real neighborhoods are ~1.1 edges/node, so
it scales fine), but ``P.dist`` broadcasts ``(N,1,D)x(1,N,D) ->
(N,N,D)`` intermediates: at N=5000, D=128 that is ~13 GB *per*
intermediate -> OOM long before the encoder strains. So here the
pairwise core is **row-chunked**, and the inherently all-pairs stats
(collapse distribution, spearman) are **pair-subsampled above a node
threshold**; each rung's table notes exact-vs-sampled. We deliberately
stop at 5000, not 50k: the point is to chart where the *tooling*
breaks, not to pretend the diagnostics scale.

Usage
-----
    py -m src.modelsv3.profile_node_scale                 # full ladder
    py -m src.modelsv3.profile_node_scale --sizes 400 800 # subset
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import psutil  # noqa: E402

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv2.layers import poincare_ops as P  # noqa: E402
from src.modelsv3.distance_ops import (  # noqa: E402
    EXACT_PAIR_NODE_CAP,
    MAX_SAMPLED_PAIRS,
    block as _block,
    chunked_topk as _chunked_topk,
    exact_pair_dists as _exact_pair_dists,
    sampled_pair_dists as _sampled_pair_dists,
)
from src.modelsv3.distance_scoring import score_from_embeddings  # noqa: E402
from src.modelsv3.eval_candidate_recall import (  # noqa: E402
    _build_encoder,
    build_query_encoder,
)
from src.training.metrics import ndcg_at_k, recall_at_k  # noqa: E402

SCRIPTS = _ROOT / "scripts"
CORPUS = _ROOT / "src" / "data" / "corpus"
RUNS = _ROOT / "runs"
V31_RUN = _ROOT / "runs" / "sweep_queryhead" / "qh1_layernorm_seed1"
NUM_GRAPHS = 12          # axis is node-size, not count: keep count modest
# EXACT_PAIR_NODE_CAP / MAX_SAMPLED_PAIRS now imported from distance_ops
# (single source of truth — same values, same chunked code path).
TAU_FRAC = 1e-4
EDGE_PREC_K = 5


# ---------------------------------------------------------------------------
# peak-RSS sampler (daemon thread; robust on Windows, no profiler hooks)
# ---------------------------------------------------------------------------

class _PeakRSS:
    def __init__(self, period: float = 0.05) -> None:
        self._proc = psutil.Process()
        self._period = period
        self._peak = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(self._period):
            try:
                self._peak = max(self._peak, self._proc.memory_info().rss)
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "_PeakRSS":
        self._t.start()
        return self

    def __exit__(self, *a) -> None:
        self._stop.set()
        self._t.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return self._peak / (1024 * 1024)


# Chunked Poincare distance core (_block / _chunked_topk /
# _sampled_pair_dists / _exact_pair_dists) is now imported verbatim from
# src.modelsv3.distance_ops above — single source of truth, identical
# code path, so node_scale_profile.json numbers are unchanged.


# ---------------------------------------------------------------------------
# BFS (single-source; used sparingly — never all-pairs at large N)
# ---------------------------------------------------------------------------

def _adj(edge_index: np.ndarray, N: int) -> list[list[int]]:
    a: list[list[int]] = [[] for _ in range(N)]
    for s, t in zip(edge_index[0], edge_index[1]):
        s, t = int(s), int(t)
        if s != t:
            a[s].append(t)
            a[t].append(s)
    return a


def _bfs(adj, src: int, N: int) -> np.ndarray:
    d = np.full(N, -1, dtype=np.int32)
    d[src] = 0
    fr = [src]
    while fr:
        nx = []
        for u in fr:
            du = d[u] + 1
            for v in adj[u]:
                if d[v] < 0:
                    d[v] = du
                    nx.append(v)
        fr = nx
    return d


# ---------------------------------------------------------------------------
# per-graph metrics (chunked / subsampled)
# ---------------------------------------------------------------------------

def _graph_metrics(emb, edge_index, c, euclidean, rng) -> dict:
    N = emb.size(0)
    exact = N <= EXACT_PAIR_NODE_CAP
    adj = _adj(edge_index, N)
    out_nb = [set() for _ in range(N)]
    for s, t in zip(edge_index[0], edge_index[1]):
        out_nb[int(s)].add(int(t))
    out_deg = np.array([len(o) for o in out_nb], dtype=np.int64)

    # --- edge_prec@5 (+ degree tercile) via chunked top-k ---
    topk = _chunked_topk(emb, EDGE_PREC_K, c, euclidean)
    hits = np.zeros(N)
    for i in range(N):
        if out_nb[i]:
            hits[i] = sum(1 for j in topk[i] if int(j) in out_nb[i]) / EDGE_PREC_K
    order = np.argsort(out_deg, kind="stable")
    lo, hi = np.array_split(order, 3)[0], np.array_split(order, 3)[2]
    edge_prec = float(hits.mean())
    edge_prec_lo = float(hits[lo].mean()) if lo.size else float("nan")
    edge_prec_hi = float(hits[hi].mean()) if hi.size else float("nan")

    # --- collapse rate + median (exact pairs if small, else sampled) ---
    if exact:
        pd = _exact_pair_dists(emb, c, euclidean)
        pair_mode = "exact"
    else:
        pd = _sampled_pair_dists(emb, c, euclidean, MAX_SAMPLED_PAIRS, rng)
        pair_mode = f"sampled({MAX_SAMPLED_PAIRS})"
    med = float(np.median(pd)) if pd.size else float("nan")
    tau = TAU_FRAC * med if med == med else 0.0
    collapse_rate = float((pd < tau).mean()) if pd.size else float("nan")

    # --- spearman(emb_dist, graph_hop) over reachable sampled pairs ---
    if exact:
        srcs = np.arange(N)
    else:
        srcs = rng.choice(N, size=min(N, 256), replace=False)
    gd, hd = [], []
    for sidx in srcs:
        hop = _bfs(adj, int(sidx), N)
        cand = np.where(hop > 0)[0]
        if cand.size == 0:
            continue
        if cand.size > 4000:
            cand = rng.choice(cand, size=4000, replace=False)
        a = emb[int(sidx)].unsqueeze(0)
        b = emb[cand]
        dd = ((a - b).norm(dim=-1).numpy() if euclidean
              else P.dist(a.expand_as(b), b, c, keepdim=False).numpy())
        gd.append(dd)
        hd.append(hop[cand].astype(np.float64))
        if sum(x.size for x in gd) >= MAX_SAMPLED_PAIRS:
            break
    spearman = float("nan")
    if gd:
        g = np.concatenate(gd)
        h = np.concatenate(hd)
        spearman = _spearman(g, h)

    return {
        "n_nodes": N,
        "pair_mode": pair_mode,
        "edge_prec@5": edge_prec,
        "edge_prec@5_high_deg": edge_prec_hi,
        "edge_prec@5_low_deg": edge_prec_lo,
        "collapse_rate": collapse_rate,
        "median_pair_dist": med,
        "spearman_emb_hop": spearman,
        "full_NN_matrix_gb": round(N * N * 128 * 4 / 1e9, 2),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den else float("nan")


def _bridge_hit5(emb, edge_index, c, euclidean, rng) -> float:
    """Sampled bridge-hit@5 (bounded #pairs; tau from sampled dists)."""
    N = emb.size(0)
    adj = _adj(edge_index, N)
    pd = _sampled_pair_dists(emb, c, euclidean, min(200_000, N * 50), rng)
    med = float(np.median(pd)) if pd.size else 1.0
    tau = TAU_FRAC * med
    hits = 0
    trials = 0
    for _ in range(20):
        u = int(rng.integers(N))
        du = _bfs(adj, u, N)
        far = np.where(du >= 3)[0]
        if far.size == 0:
            continue
        v = int(far[rng.integers(far.size)])
        trials += 1
        dv = _bfs(adj, v, N)
        on_path = set(int(w) for w in np.where(
            (du >= 0) & (dv >= 0) & (du + dv == du[v]))[0]) - {u, v}
        if not on_path:
            continue
        ut = P.logmap0(emb[u].unsqueeze(0), c)
        vt = P.logmap0(emb[v].unsqueeze(0), c)
        m = P.expmap0((ut + vt) / 2, c).squeeze(0)
        if euclidean:
            m = (emb[u] + emb[v]) / 2
        dist = ((emb - m).norm(dim=-1).numpy() if euclidean
                else P.dist(emb, m.unsqueeze(0).expand_as(emb), c,
                            keepdim=False).numpy())
        order = np.argsort(dist)
        kept = [int(i) for i in order
                if i not in (u, v) and dist[int(i)] >= tau][:5]
        if any(w in on_path for w in kept):
            hits += 1
    return hits / trials if trials else float("nan")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _ensure_corpus(max_nodes: int) -> Path:
    cdir = CORPUS / f"real_domain_eval_nodescale_{max_nodes}"
    if cdir.is_dir() and any(cdir.glob("graph_*.npz")):
        return cdir
    print(f"[nodescale] export delocalized N<= {max_nodes} "
          f"({NUM_GRAPHS} graphs)", flush=True)
    cmd = [
        sys.executable, "neo4j_eval_export.py", "export",
        "--config", "kettle_config.yaml",
        "--out", str(Path("..") / "src" / "data" / "corpus" / cdir.name),
        "--num-graphs", str(NUM_GRAPHS), "--max-nodes", str(max_nodes),
        "--tasks-per-graph", "3", "--seed", "0", "--sampler", "delocalized",
    ]
    log = RUNS / f"nodescale_export_{max_nodes}.log"
    with open(log, "w") as f:
        subprocess.call(cmd, cwd=str(SCRIPTS), stdout=f,
                        stderr=subprocess.STDOUT)
    return cdir


def _load_model(ds: CorpusDataset):
    cfg = json.loads((V31_RUN / "summary.json").read_text())["config"]
    enc = _build_encoder(cfg, ds)
    enc.load_state_dict(torch.load(V31_RUN / "encoder.pt", map_location="cpu"))
    enc.eval()
    qe = build_query_encoder(cfg, ds)
    qe.load_state_dict(torch.load(V31_RUN / "query_encoder.pt",
                                  map_location="cpu"))
    qe.eval()
    c = getattr(enc, "c", torch.tensor(float(cfg.get("curvature", 1.0))))
    return enc, qe, cfg["model"] == "euclidean", c


def run(sizes: list[int]) -> int:
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for M in sizes:
        cdir = _ensure_corpus(M)
        if not (cdir.is_dir() and any(cdir.glob("graph_*.npz"))):
            print(f"[nodescale] M={M}: export failed; stop ladder", flush=True)
            break
        ds = CorpusDataset(corpus_dir=str(cdir), split="all", split_seed=0,
                           include_tasks={2})
        enc, qe, euc, c = _load_model(ds)
        print(f"[nodescale] M={M}: {len(ds)} samples "
              f"{len(set(g for g, _ in ds.index))} graphs", flush=True)

        enc_times, mets = [], []
        ndcg10 = ndcg20 = r50 = r100 = bh5 = 0.0
        n = 0
        with _PeakRSS() as peak:
            with torch.no_grad():
                emb_cache: dict[int, torch.Tensor] = {}
                for i in range(len(ds)):
                    gi, _ = ds.index[i]
                    s = ds[i]
                    if gi not in emb_cache:
                        t0 = time.perf_counter()
                        o = enc(s.x, s.edge_index, s.edge_type,
                                s.edge_descriptor,
                                node_descriptor=s.node_descriptor)
                        enc_times.append(time.perf_counter() - t0)
                        emb_cache[gi] = o.node_embeddings.detach()
                        mets.append(_graph_metrics(
                            emb_cache[gi], s.edge_index.numpy(), c, euc, rng))
                        bh5 += _bridge_hit5(emb_cache[gi],
                                            s.edge_index.numpy(), c, euc, rng)
                    emb = emb_cache[gi]
                    sc = score_from_embeddings(emb, qe(s.query), c=c,
                                               euclidean=euc)
                    ndcg10 += ndcg_at_k(sc, s.labels, 10)
                    ndcg20 += ndcg_at_k(sc, s.labels, 20)
                    r50 += recall_at_k(sc, s.labels, 50)
                    r100 += recall_at_k(sc, s.labels, 100)
                    n += 1
            peak_mb = peak.peak_mb
        ng = len(mets)

        def avg(key):
            v = [m[key] for m in mets if m[key] == m[key]]
            return float(np.mean(v)) if v else float("nan")

        row = {
            "max_nodes": M,
            "graphs": ng,
            "mean_actual_nodes": float(np.mean([m["n_nodes"] for m in mets])),
            "encode_time_s_mean": float(np.mean(enc_times)),
            "peak_rss_mb": round(peak_mb, 1),
            "full_NN_matrix_gb": mets[0]["full_NN_matrix_gb"],
            "pair_mode": mets[0]["pair_mode"],
            "ndcg@10": ndcg10 / n, "ndcg@20": ndcg20 / n,
            "recall@50": r50 / n, "recall@100": r100 / n,
            "edge_prec@5": avg("edge_prec@5"),
            "edge_prec@5_high_deg": avg("edge_prec@5_high_deg"),
            "edge_prec@5_low_deg": avg("edge_prec@5_low_deg"),
            "bridge_hit@5": bh5 / ng,
            "collapse_rate": avg("collapse_rate"),
            "spearman_emb_hop": avg("spearman_emb_hop"),
        }
        rows.append(row)
        print(f"[nodescale]   N~{row['mean_actual_nodes']:.0f}  "
              f"enc={row['encode_time_s_mean']:.2f}s  "
              f"peakRSS={row['peak_rss_mb']:.0f}MB  "
              f"ndcg@10={row['ndcg@10']:.4f}  "
              f"edge_p@5={row['edge_prec@5']:.4f}  "
              f"({row['pair_mode']})", flush=True)

    out = RUNS / "node_scale_profile.json"
    out.write_text(json.dumps(rows, indent=2))
    _print(rows)
    print(f"\n[nodescale] -> {out}")
    return 0


def _print(rows: list[dict]) -> None:
    print()
    print("=" * 110)
    print("v3.1 neighborhood-SIZE scaling  (frozen encoder + qh1; "
          "hardened/delocalized; chunked dist,")
    print(f"pair-subsampled above {EXACT_PAIR_NODE_CAP} nodes)")
    print("=" * 110)
    cols = [
        ("max_nodes", "maxN", "{:.0f}"),
        ("mean_actual_nodes", "N~", "{:.0f}"),
        ("encode_time_s_mean", "enc_s", "{:.2f}"),
        ("peak_rss_mb", "peakMB", "{:.0f}"),
        ("full_NN_matrix_gb", "NNmat_GB", "{:.2f}"),
        ("ndcg@10", "ndcg@10", "{:.4f}"),
        ("recall@50", "r@50", "{:.4f}"),
        ("recall@100", "r@100", "{:.4f}"),
        ("edge_prec@5", "edgeP@5", "{:.4f}"),
        ("bridge_hit@5", "bridge@5", "{:.4f}"),
        ("collapse_rate", "collapse", "{:.5f}"),
        ("spearman_emb_hop", "spearman", "{:.4f}"),
    ]
    print("  " + " ".join(f"{lbl:>10}" for _, lbl, _ in cols))
    for r in rows:
        cells = []
        for name, _, fmt in cols:
            v = r.get(name)
            cells.append(fmt.format(v) if isinstance(v, (int, float))
                         and v == v else "n/a")
        print("  " + " ".join(f"{c:>10}" for c in cells))
    print("\n  NNmat_GB = the (N,N,128) f32 matrix the SHIPPED eval scripts "
          "would allocate (chunked away here).")
    print("  That, not the O(edges) encoder, is the tooling bottleneck. "
          "Ladder stops at 5000 by design.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[400, 800, 1500, 3000, 5000])
    args = ap.parse_args()
    return run(args.sizes)


if __name__ == "__main__":
    sys.exit(main())
