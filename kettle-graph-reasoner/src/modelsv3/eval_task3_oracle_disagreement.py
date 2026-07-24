r"""Phase 1 - task-3 oracle-disagreement diagnostic (read-only).

Hypothesis: task-3 (multi-hop) candidates that are oracle-good but
model-low are distinguished by GRAPH hop-distance from the anchor -- a
signal the retriever embedding + v2 cannot see. Task-3 relevance is
literally BFS-hop-decay from an anchor (task_generator.py:374-469:
``0.7**hop * (1+rarity) * branch_penalty``, cutoff <0.15), but the
task-3 query does NOT encode the anchor. This diagnostic tests, on the
exact deployed candidate set (the qh1 retriever's top-C), whether hop is
the missing discriminative axis and whether a leak-free anchor proxy
(retriever top-1) tracks the true anchor well enough to seed a Phase-2
structural feature.

The TRUE anchor is read READ-ONLY from the npz (``task_{j}_anchor_row``,
corpus_builder.py:246) -- no corpus regen, no proxy needed for the
diagnosis itself. Pure measurement; ASCII only (Windows cp1252).

Pre-specified PASS gate (auto-gates Phase 2):
  (a) mean spearman(label, -hop) > 0.25 AND clearly > model's
      (gap > across-graph std of the per-sample gap), AND
  (b) mean missed relevant label-mass at hop>=2 > 0.30, AND
  (c) median per-graph spearman(true-hop, proxy-hop) > 0.60.
PASS => Phase 2 (hop-structural reranker term). FAIL => reframed lever.

Usage
-----
    py -m src.modelsv3.eval_task3_oracle_disagreement \
        --run runs/v3.1-baseline-hyp-h128-l4-seed1 --task 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv3.distance_scoring import score_from_embeddings  # noqa: E402
from src.modelsv3.eval_candidate_recall import (  # noqa: E402
    _build_encoder,
    build_query_encoder,
)
from src.modelsv3.eval_geom_graph_disagreement import _spearman  # noqa: E402
from src.modelsv3.eval_retrieval_midpoint import _bfs_hop_matrix  # noqa: E402

TASK = 3
TOPC = 50
REL = 0.15          # task_generator.py:459 hard label cutoff
ALPHA = 0.7         # task_generator.py:403 distance-decay base
DEG_COL = 16        # feature_encoder.py structural [16] = log total degree
# PASS thresholds (plan, pre-specified)
T_LABEL_CORR = 0.25
T_MISSED = 0.30
T_PROXY = 0.60


def _load_retriever(retr: Path, dataset: CorpusDataset):
    """Same retriever the deployed reranker uses (reranker_v32 semantics):
    frozen v3.1 encoder + the per-task qh1 head, read-only."""
    cfg = json.loads((retr / "summary.json").read_text())["config"]
    enc = _build_encoder(cfg, dataset)
    enc.load_state_dict(torch.load(retr / "encoder.pt", map_location="cpu"))
    enc.eval()
    qe = build_query_encoder(cfg, dataset)
    qe.load_state_dict(torch.load(retr / "query_encoder.pt",
                                  map_location="cpu"))
    qe.eval()
    c_val = getattr(enc, "c",
                    torch.tensor(float(cfg.get("curvature", 1.0))))
    return enc, qe, c_val, cfg["model"] == "euclidean"


def _hop_for_rank(hop_row: np.ndarray) -> np.ndarray:
    """-hop signal where unreachable (-1) ranks FARTHEST (not closest)."""
    reach = hop_row[hop_row >= 0]
    far = (int(reach.max()) + 1) if reach.size else 1
    h = np.where(hop_row < 0, far, hop_row).astype(np.float64)
    return -h


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    """OLS R^2 of y ~ [X, 1]. Upper-bounds how much the regressors
    explain label; 1-R^2 upper-bounds the (unrecoverable) rarity term."""
    A = np.column_stack([X, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    sstot = float(((y - y.mean()) ** 2).sum())
    if sstot < 1e-12:
        return float("nan")
    return 1.0 - float((resid ** 2).sum()) / sstot


def run(retr: Path, corpus: str, split: str, split_seed: int,
        topc: int, out_path: Path) -> dict:
    ds = CorpusDataset(corpus_dir=corpus, split=split,
                       split_seed=split_seed, include_tasks={TASK})
    enc, qe, c_val, euclidean = _load_retriever(retr, ds)
    print(f"[evalT3] {len(ds)} task-{TASK} {split} samples  "
          f"retriever={retr.name}")

    emb_cache: dict[int, torch.Tensor] = {}
    per: list[dict] = []
    n_skip = 0
    n_anchor0_excl = 0
    n_anchor_argmax_mismatch = 0
    deg_min, deg_max = 1e9, -1e9

    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            gi, j = ds.index[i]
            with np.load(ds.files[gi]) as npz:
                anchor = int(npz[f"task_{j}_anchor_row"])

            if gi not in emb_cache:
                emb_cache[gi] = enc(
                    s.x, s.edge_index, s.edge_type, s.edge_descriptor,
                    node_descriptor=s.node_descriptor,
                ).node_embeddings.detach()
            emb = emb_cache[gi]
            N = emb.size(0)

            labels = s.labels.cpu().numpy()
            if (labels >= REL).sum() < 1 or np.unique(labels).size < 2:
                n_skip += 1
                continue
            lbl_argmax = int(labels.argmax())
            # anchor==0 ambiguity (corpus_builder.py:97 maps missing->0):
            # exclude only if it disagrees with the label hop-0 peak.
            if anchor == 0 and lbl_argmax != 0:
                n_anchor0_excl += 1
                continue
            if lbl_argmax != anchor:
                n_anchor_argmax_mismatch += 1

            scores = score_from_embeddings(
                emb, qe(s.query), c=c_val, euclidean=euclidean)
            sc = scores.cpu().numpy()
            C = min(topc, N)
            cand = np.argsort(-sc)[:C]

            hop = _bfs_hop_matrix(s.edge_index, N)   # (N,N), -1 unreach
            hop_true = hop[anchor]
            proxy = int(cand[np.argmax(sc[cand])])    # retriever top-1
            hop_proxy = hop[proxy]

            deg = s.x[:, DEG_COL].cpu().numpy()
            deg_min = min(deg_min, float(deg.min()))
            deg_max = max(deg_max, float(deg.max()))

            lab_c = labels[cand]
            mod_c = sc[cand]
            sig_c = _hop_for_rank(hop_true[cand])     # -hop, unreach far
            if np.unique(lab_c).size < 2:
                n_skip += 1
                continue
            sp_lab = _spearman(lab_c, sig_c)
            sp_mod = _spearman(mod_c, sig_c)
            if sp_lab != sp_lab or sp_mod != sp_mod:
                n_skip += 1
                continue

            # (b) missed relevant label-mass at hop>=2 (model vs label top-10)
            mtop = set(cand[np.argsort(-mod_c)[:10]].tolist())
            rel_idx = [int(n) for n in cand
                       if labels[n] >= REL and hop_true[n] >= 2]
            if rel_idx:
                tot = float(sum(labels[n] for n in rel_idx))
                miss = float(sum(labels[n] for n in rel_idx
                                 if n not in mtop))
                missed_mass = miss / tot if tot > 1e-9 else 0.0
            else:
                missed_mass = float("nan")

            # label decomposition: label ~ [0.7**hop, branch_penalty]
            hclip = np.where(hop_true < 0, 0.0,
                             ALPHA ** np.maximum(hop_true, 0))
            # branch_penalty ~ 1/max(log(deg),1); x[:,16] IS log(deg)
            bp = 1.0 / np.maximum(deg, 1.0)
            r2 = _r2(labels.astype(np.float64),
                     np.column_stack([hclip, bp]))

            # (c) proxy reliability: spearman over reachable-by-both
            both = (hop_true >= 0) & (hop_proxy >= 0)
            sp_proxy = (_spearman(hop_true[both].astype(np.float64),
                                  hop_proxy[both].astype(np.float64))
                        if both.sum() >= 2 else float("nan"))

            per.append({
                "sp_label_hop": sp_lab,
                "sp_model_hop": sp_mod,
                "gap": sp_lab - sp_mod,
                "missed_mass_hop_ge2": missed_mass,
                "label_r2_hop_branch": r2,
                "proxy_true_hop_spearman": sp_proxy,
                "n_cand": int(C),
                "n_reach_from_anchor": int((hop_true >= 0).sum()),
            })

    deg_ok = deg_min >= -1e-6  # log-degree must be >= 0
    res = _aggregate(per)
    res.update({
        "retriever_run": str(retr), "split": split, "task": TASK,
        "topc": topc, "n_eval": len(per), "n_skipped": n_skip,
        "n_anchor0_excluded": n_anchor0_excl,
        "n_anchor_argmax_mismatch": n_anchor_argmax_mismatch,
        "deg_feature_range": [deg_min, deg_max],
        "deg_feature_nonneg": bool(deg_ok),
    })
    res["gate"] = _gate(res)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    _print(res, out_path)
    return res


def _ms(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    return (float(statistics.mean(xs)),
            float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0)


def _aggregate(per: list[dict]) -> dict:
    if not per:
        return {"empty": True}
    lab_m, lab_s = _ms([p["sp_label_hop"] for p in per])
    mod_m, mod_s = _ms([p["sp_model_hop"] for p in per])
    gap_m, gap_s = _ms([p["gap"] for p in per])
    miss_m, miss_s = _ms([p["missed_mass_hop_ge2"] for p in per])
    r2_m, r2_s = _ms([p["label_r2_hop_branch"] for p in per])
    px = [p["proxy_true_hop_spearman"] for p in per
          if p["proxy_true_hop_spearman"] == p["proxy_true_hop_spearman"]]
    px_med = float(statistics.median(px)) if px else float("nan")
    return {
        "sp_label_hop_mean": lab_m, "sp_label_hop_std": lab_s,
        "sp_model_hop_mean": mod_m, "sp_model_hop_std": mod_s,
        "D_label_minus_model_mean": gap_m, "D_std": gap_s,
        "missed_mass_hop_ge2_mean": miss_m, "missed_mass_hop_ge2_std":
        miss_s,
        "label_r2_hop_branch_mean": r2_m, "label_r2_hop_branch_std": r2_s,
        "proxy_true_hop_spearman_median": px_med,
        "n_with_proxy": len(px),
    }


def _gate(r: dict) -> dict:
    a = (r["sp_label_hop_mean"] > T_LABEL_CORR
         and (r["D_label_minus_model_mean"] > r["D_std"])
         and r["sp_label_hop_mean"] > r["sp_model_hop_mean"])
    b = r["missed_mass_hop_ge2_mean"] > T_MISSED
    cc = r["proxy_true_hop_spearman_median"] > T_PROXY
    pass_ = bool(a and b and cc)
    if pass_:
        branch = ("PASS -> Phase 2: hop-structural reranker term "
                  "(leak-free retriever-top-1 seed viable).")
    elif not cc:
        branch = ("FAIL (proxy unreliable) -> do NOT build struct; "
                  "reframe as retriever/query-head lever (task-3 qh1 "
                  "missed the WS2 bar by 0.145; qh2/qh3 never tried).")
    elif not a and r["label_r2_hop_branch_mean"] < 0.5:
        branch = ("FAIL (rarity dominates: low label~hop+branch R^2) -> "
                  "hop is the wrong signal; the separate lever is a "
                  "path-edge-type-rarity proxy. Do not build hop.")
    else:
        branch = ("FAIL (hop weak / model already tracks hop) -> "
                  "ordering failure not hop-explained; reframe as "
                  "retriever/query-head lever.")
    return {"cond_a_label_hop_signal": bool(a),
            "cond_b_missed_mass": bool(b),
            "cond_c_proxy_reliable": bool(cc),
            "phase1_pass": pass_, "branch": branch,
            "thresholds": {"label_corr": T_LABEL_CORR,
                           "missed_mass": T_MISSED, "proxy": T_PROXY}}


def _print(r: dict, out: Path) -> None:
    print()
    print("=" * 92)
    print(f"Phase 1 - task-{TASK} oracle-disagreement diagnostic")
    print("=" * 92)
    if r.get("empty"):
        print("  no eligible samples (all skipped). Check the retriever "
              "run / split.")
        return
    print(f"  n_eval={r['n_eval']}  skipped={r['n_skipped']}  "
          f"anchor0_excl={r['n_anchor0_excluded']}  "
          f"anchor!=argmax(label)={r['n_anchor_argmax_mismatch']}  "
          f"deg_nonneg={r['deg_feature_nonneg']}")
    print(f"  spearman(label,-hop) = {r['sp_label_hop_mean']:+.3f} "
          f"+-{r['sp_label_hop_std']:.3f}   (threshold > {T_LABEL_CORR})")
    print(f"  spearman(model,-hop) = {r['sp_model_hop_mean']:+.3f} "
          f"+-{r['sp_model_hop_std']:.3f}")
    print(f"  D = label - model    = {r['D_label_minus_model_mean']:+.3f} "
          f"  (need > D_std {r['D_std']:.3f})")
    print(f"  missed relevant mass @ hop>=2 = "
          f"{r['missed_mass_hop_ge2_mean']:.3f} "
          f"+-{r['missed_mass_hop_ge2_std']:.3f}  (threshold > {T_MISSED})")
    print(f"  label ~ [0.7**hop, branch] R^2 = "
          f"{r['label_r2_hop_branch_mean']:.3f}  "
          f"(1-R^2 = {1.0 - r['label_r2_hop_branch_mean']:.3f} upper-bounds "
          f"the unrecoverable rarity term)")
    print(f"  proxy(top-1) vs true-anchor hop spearman, median = "
          f"{r['proxy_true_hop_spearman_median']:.3f}  "
          f"(threshold > {T_PROXY}; n={r['n_with_proxy']})")
    g = r["gate"]
    print(f"\n  cond (a) label-hop signal : {g['cond_a_label_hop_signal']}")
    print(f"  cond (b) missed-mass>{T_MISSED} : {g['cond_b_missed_mass']}")
    print(f"  cond (c) proxy reliable   : {g['cond_c_proxy_reliable']}")
    print(f"\n  PHASE-1 {'PASS' if g['phase1_pass'] else 'FAIL'} : "
          f"{g['branch']}")
    print(f"\n  results: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever-run", type=str,
                    default="runs/sweep_taskdiversity/task3_seed0",
                    help="the per-task qh1 retriever the reranker uses")
    ap.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--topc", type=int, default=TOPC)
    ap.add_argument("--task", type=int, default=TASK,
                    help="fixed to 3 (multi-hop); accepted for parity")
    ap.add_argument("--run", type=str,
                    default="runs/v3.1-baseline-hyp-h128-l4-seed1",
                    help="output dir (writes task3_oracle_disagreement.json)")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if int(a.task) != TASK:
        print(f"[evalT3] WARNING: --task {a.task} ignored; this "
              f"diagnostic is task-{TASK} only.")
    out = Path(a.out) if a.out else (
        Path(a.run) / "task3_oracle_disagreement.json")
    run(Path(a.retriever_run), a.corpus, a.split, a.split_seed,
        a.topc, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
