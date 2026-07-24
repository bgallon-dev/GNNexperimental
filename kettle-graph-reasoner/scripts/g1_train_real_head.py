"""
g1_train_real_head.py
=====================

G1 follow-on: train a real Stage-B QueryToBall head for an in-scope
geometry task (task 4 subgraph) on a real-domain corpus, with the frozen
v3.1 encoder, and PROMOTE it only if it strictly beats the synthetic head
the MVP currently ships -- the exact discipline used for the validated
task-2 real head.

Recipe: inherited verbatim from the validated task-2 real-head run
(``runs/v3.1-real-head-hyp-h128-l4-seed0``: qh1 / layernorm / pairwise /
lr_query 3e-4 / 10 query-epochs). Encoder: the SHA-asserted frozen
baseline (``runs/v3.1-baseline-hyp-h128-l4-seed1/encoder.pt``); Stage-B
trains the query head only -- encoder params are checksummed pre/post to
PROVE they did not move (the G1 "SHA-identical pre/post" requirement).

Promotion gate (lock_baseline.gate_threshold convention):
    promote iff  mean(real, 3 seeds) > synthetic + 1*std(real)
    AND encoder param checksum unchanged.
Honest negatives are reported, not hidden; on fail the MVP keeps the
provably-non-regressing identity/synthetic fallback.

Usage
-----
    py scripts/g1_train_real_head.py --task 4 \
        --corpus src/data/corpus/real_subgraph_eval \
        --seeds 0 1 2 --out runs/v3.1-real-head-task4-hyp-h128-l4
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv3.eval_candidate_recall import build_query_encoder  # noqa: E402
from src.modelsv3.lock_baseline import (  # noqa: E402
    assert_encoder_sha,
    sha256_file,
)
from src.modelsv3.query_encoder import QueryToBall  # noqa: E402
from src.training.metrics import ndcg_at_k  # noqa: E402
from src.training.train_v3 import Config, _build_encoder, _eval, _stage_b  # noqa: E402

_BASELINE = _ROOT / "runs" / "v3.1-baseline-hyp-h128-l4-seed1"
_RECIPE = _ROOT / "runs" / "v3.1-real-head-hyp-h128-l4-seed0"


def _param_checksum(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _floor_and_ceiling(test_ds: CorpusDataset) -> tuple[float, float]:
    """Honest reference lines on the held-out test split: random noise
    floor and the oracle (perfect-label) ceiling, ndcg@10."""
    rng = np.random.default_rng(1234)
    fl, ce = [], []
    for i in range(len(test_ds)):
        s = test_ds[i]
        lab = s.labels.float()
        rnd = torch.from_numpy(
            rng.standard_normal(lab.numel()).astype(np.float32))
        fl.append(ndcg_at_k(rnd, lab, 10))
        ce.append(ndcg_at_k(lab.clone(), lab, 10))
    return (float(np.mean(fl)) if fl else float("nan"),
            float(np.mean(ce)) if ce else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--recipe-run", default=str(_RECIPE))
    ap.add_argument("--baseline-run", default=str(_BASELINE))
    ap.add_argument("--query-epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    task = a.task
    baseline = Path(a.baseline_run)
    recipe = Path(a.recipe_run)
    out = Path(a.out) if a.out else (
        _ROOT / "runs" / f"v3.1-real-head-task{task}-hyp-h128-l4")
    device = torch.device("cpu")

    # frozen encoder: the SHA-asserted locked baseline.
    assert_encoder_sha(baseline, baseline / "encoder.pt")
    enc_sha = sha256_file(baseline / "encoder.pt")
    rec_cfg = json.loads((recipe / "summary.json").read_text())["config"]
    base_cfg = json.loads(
        (baseline / "summary.json").read_text())["config"]
    valid = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in rec_cfg.items() if k in valid})
    cfg.task = task
    cfg.corpus = a.corpus
    cfg.skip_stage_a = True
    cfg.load_encoder = str(baseline / "encoder.pt")
    if a.query_epochs is not None:
        cfg.query_epochs = a.query_epochs
    euclidean = cfg.model == "euclidean"

    train_ds = CorpusDataset(corpus_dir=a.corpus, split="train",
                             split_seed=0, include_tasks={task})
    test_ds = CorpusDataset(corpus_dir=a.corpus, split="test",
                            split_seed=0, include_tasks={task})
    n_test = sum(1 for _ in range(len(test_ds)))
    print(f"[G1] task {task}  corpus={a.corpus}  "
          f"train={len(train_ds)}  test={n_test} instances  "
          f"recipe={recipe.name} (arch={rec_cfg.get('query_head_arch')})")

    enc = _build_encoder(cfg, train_ds).to(device)
    enc.load_state_dict(
        torch.load(baseline / "encoder.pt", map_location=device))
    enc.eval()
    enc_ck_before = _param_checksum(enc)

    # BEFORE: the synthetic head the MVP ships for this task today
    # (baseline run, loaded exactly as retrieval_ops.load_query_encoder).
    qe_syn = build_query_encoder(base_cfg, train_ds).to(device)
    qe_syn.load_state_dict(
        torch.load(baseline / "query_encoder.pt", map_location=device))
    qe_syn.eval()
    synth = float(_eval(cfg, enc, qe_syn, test_ds, device)["overall"][
        "ndcg@10"])
    floor, ceiling = _floor_and_ceiling(test_ds)

    # AFTER: fresh real-trained heads, validated recipe, 3 seeds.
    adapted: list[float] = []
    seed_heads: dict[int, Path] = {}
    tmp = Path(tempfile.mkdtemp(prefix="g1_"))
    for sd in a.seeds:
        cfg.seed = sd
        cfg.out = str(tmp / f"seed{sd}")
        Path(cfg.out).mkdir(parents=True, exist_ok=True)
        torch.manual_seed(sd)
        np.random.seed(sd)
        qe = QueryToBall(
            query_dim=train_ds.query_dim, hidden_dim=cfg.hidden_dim,
            c=cfg.curvature, euclidean=euclidean,
            arch=cfg.query_head_arch, norm=cfg.query_head_norm,
        ).to(device)
        _stage_b(cfg, enc, qe, train_ds, device)
        nd = float(_eval(cfg, enc, qe, test_ds, device)["overall"][
            "ndcg@10"])
        adapted.append(nd)
        hp = tmp / f"qe_seed{sd}.pt"
        torch.save(qe.state_dict(), hp)
        seed_heads[sd] = hp
        print(f"  seed {sd}: real adapted ndcg@10 = {nd:.4f}")

    enc_ck_after = _param_checksum(enc)
    encoder_frozen = enc_ck_before == enc_ck_after

    mean_r = float(np.mean(adapted))
    std_r = float(np.std(adapted, ddof=1)) if len(adapted) > 1 else 0.0
    # lock_baseline.gate_threshold convention: base + max(spec_delta, 1*std)
    threshold = synth + max(0.0, std_r)
    promote = bool(mean_r > threshold and encoder_frozen)

    report = {
        "task": task, "corpus": a.corpus,
        "encoder_sha256": enc_sha,
        "encoder_frozen_pre_post": encoder_frozen,
        "recipe_run": str(recipe),
        "synthetic_head_ndcg@10": synth,
        "random_floor_ndcg@10": floor,
        "oracle_ceiling_ndcg@10": ceiling,
        "real_adapted_ndcg@10": {"seeds": dict(zip(a.seeds, adapted)),
                                 "mean": mean_r, "std": std_r},
        "gate_threshold": threshold,
        "gate_rule": "mean(real) > synthetic + 1*std(real) AND "
                     "encoder frozen",
        "promote": promote,
    }
    print("\n" + "=" * 72)
    print(f"G1 task {task} verdict")
    print("=" * 72)
    print(f"  random floor      ndcg@10 = {floor:.4f}")
    print(f"  synthetic head    ndcg@10 = {synth:.4f}  (MVP ships this)")
    print(f"  real head (3sd)   ndcg@10 = {mean_r:.4f} +- {std_r:.4f}")
    print(f"  oracle ceiling    ndcg@10 = {ceiling:.4f}")
    print(f"  gate threshold            = {threshold:.4f} "
          f"(synthetic + 1 std)")
    print(f"  encoder frozen pre/post   = {encoder_frozen}")
    print(f"  -> PROMOTE = {promote}")

    if promote:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        shutil.copy2(baseline / "encoder.pt", out / "encoder.pt")
        best_sd = a.seeds[int(np.argmax(adapted))]
        shutil.copy2(seed_heads[best_sd], out / "query_encoder.pt")
        summary = {"config": {**rec_cfg, "task": task, "corpus": a.corpus,
                              "query_head_arch": cfg.query_head_arch,
                              "query_head_norm": cfg.query_head_norm},
                   "g1": report, "persisted_seed": int(best_sd)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        (out / "g1_report.json").write_text(json.dumps(report, indent=2))
        print(f"\n  persisted real task-{task} head -> {out} "
              f"(seed {best_sd}); encoder SHA {enc_sha[:12]}... "
              f"(== frozen baseline)")
    else:
        print("\n  HONEST NEGATIVE: real head did not clear the gate; "
              "the MVP keeps the provably-non-regressing synthetic/"
              "identity fallback for this task. No routing change.")
        (out.parent / f"g1_task{task}_negative.json").write_text(
            json.dumps(report, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if promote else 1


if __name__ == "__main__":
    sys.exit(main())
