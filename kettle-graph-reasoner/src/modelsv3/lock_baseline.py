r"""v3.1 Phase 1.0/1.1 — baseline lock + noise floor.

Freezes the uploaded ``h128_l4_seed1`` checkpoint as the named v3.1
comparison target. This script *only copies and hashes* — it never
retrains and never writes back into the source run. The "freeze" is
enforced by convention plus a SHA-256 assertion the later phases check.

What it produces
----------------
``<out>/`` containing copies of
``encoder.pt query_encoder.pt summary.json intrinsic_eval.json
collapse.json`` from the source run, plus ``baseline_manifest.json``:

    {
      "name": "v3.1-baseline-hyp-h128-l4-seed1",
      "source_run": "...",
      "encoder_sha256": "...",
      "query_encoder_sha256": "...",
      "config": { ...copied verbatim from summary.json["config"]... },
      "frozen_metrics": { final_val + corpus-wide intrinsic },
      "noise_floor": { ndcg@10 / ndcg@20 / edge_prec@5 mean+std
                       over the seed family h128_l4_seed{0,1,2} }
    }

Every Phase-2/3/4 gate is stated as
``baseline_mean + max(spec_delta, 1 * combined_std)`` — use
:func:`gate_threshold` (importable) so the rule lives in one place.

Usage
-----
    py -m src.modelsv3.lock_baseline \
        --source runs/sweep_arch_hyp/h128_l4_seed1 \
        --out    runs/v3.1-baseline-hyp-h128-l4-seed1

``--seeds-family`` defaults to every sibling ``h128_l4_seed*`` dir of
``--source`` (the noise-floor cohort). Pass paths explicitly to override.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path

_COPY_FILES = (
    "encoder.pt",
    "query_encoder.pt",
    "summary.json",
    "intrinsic_eval.json",
    "collapse.json",
)


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# noise floor
# ---------------------------------------------------------------------------

def _summarize(vals: list[float]) -> dict:
    clean = [float(v) for v in vals if v == v]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}
    return {
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
        "n": len(clean),
        "values": clean,
    }


def compute_noise_floor(seed_dirs: list[Path]) -> dict:
    """mean/std of the gate-relevant metrics across the seed family.

    Pulls ``final_val.overall.ndcg@{10,20}`` and
    ``intrinsic_val_graph0.nn_edge_precision@5.mean_precision`` from each
    run's ``summary.json``. The single-graph intrinsic proxy is what the
    training pipeline writes per run; the corpus-wide value (from
    ``intrinsic_eval.json``) is recorded separately in ``frozen_metrics``.
    """
    ndcg10: list[float] = []
    ndcg20: list[float] = []
    edge_prec5: list[float] = []
    used: list[str] = []
    for d in seed_dirs:
        sp = d / "summary.json"
        if not sp.exists():
            continue
        with open(sp, "r") as f:
            s = json.load(f)
        fv = s.get("final_val", {}).get("overall", {})
        if "ndcg@10" in fv:
            ndcg10.append(fv["ndcg@10"])
        if "ndcg@20" in fv:
            ndcg20.append(fv["ndcg@20"])
        ep = s.get("intrinsic_val_graph0", {}).get("nn_edge_precision@5", {})
        if "mean_precision" in ep:
            edge_prec5.append(ep["mean_precision"])
        used.append(str(d))
    return {
        "seed_family": used,
        "n_seeds": len(used),
        "ndcg@10": _summarize(ndcg10),
        "ndcg@20": _summarize(ndcg20),
        "intrinsic_edge_prec@5": _summarize(edge_prec5),
    }


def gate_threshold(noise_floor: dict, metric: str, spec_delta: float) -> float:
    """The one place the gate rule lives:
    ``baseline_mean + max(spec_delta, 1 * std)``.

    ``metric`` is a key under ``noise_floor`` (e.g. ``"ndcg@10"``).
    ``spec_delta`` is the absolute improvement the v3.1 spec asks for
    (e.g. 0.52 - baseline_mean). Pass the *delta*, not the target.
    """
    block = noise_floor[metric]
    mean = block["mean"]
    std = block["std"] if block["std"] == block["std"] else 0.0
    return mean + max(spec_delta, std)


# ---------------------------------------------------------------------------
# frozen metrics
# ---------------------------------------------------------------------------

def _frozen_metrics(source: Path) -> dict:
    out: dict = {}
    sp = source / "summary.json"
    with open(sp, "r") as f:
        s = json.load(f)
    out["final_val_overall"] = s.get("final_val", {}).get("overall", {})
    out["intrinsic_val_graph0"] = s.get("intrinsic_val_graph0", {})
    out["n_params_encoder"] = s.get("n_params_encoder")
    out["n_params_query"] = s.get("n_params_query")

    iep = source / "intrinsic_eval.json"
    if iep.exists():
        with open(iep, "r") as f:
            ie = json.load(f)
        summ = ie.get("summary", {})
        out["intrinsic_corpus"] = {
            "edge_precision_at_k_mean": summ.get("edge_precision_at_k", {}).get("mean"),
            "silhouette_mean": summ.get("silhouette", {}).get("mean"),
            "label_purity_at_k_mean": summ.get("label_purity_at_k", {}).get("mean"),
            "random_baseline_edge_prec_mean": summ.get("random_baseline_edge_prec_mean"),
            "random_baseline_label_purity_mean": summ.get("random_baseline_label_purity_mean"),
        }
    return out


# ---------------------------------------------------------------------------
# manifest helpers (importable by later phases / evals)
# ---------------------------------------------------------------------------

def manifest_path(baseline_dir: Path) -> Path:
    return Path(baseline_dir) / "baseline_manifest.json"


def load_manifest(baseline_dir: Path) -> dict:
    with open(manifest_path(baseline_dir), "r") as f:
        return json.load(f)


def assert_encoder_sha(baseline_dir: Path, encoder_pt: Path) -> None:
    """Raise if ``encoder_pt`` does not match the locked baseline hash.
    Used by export/index scripts that must prove they indexed the asset
    the manifest describes."""
    man = load_manifest(baseline_dir)
    got = sha256_file(Path(encoder_pt))
    want = man["encoder_sha256"]
    if got != want:
        raise ValueError(
            f"encoder SHA mismatch: {encoder_pt} has {got[:12]}…, "
            f"baseline manifest expects {want[:12]}…"
        )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def lock_baseline(
    source: Path, out: Path, name: str, seed_dirs: list[Path]
) -> dict:
    source = Path(source)
    out = Path(out)
    if not source.is_dir():
        raise FileNotFoundError(f"source run not found: {source}")
    out.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for fn in _COPY_FILES:
        src_f = source / fn
        if not src_f.exists():
            raise FileNotFoundError(f"expected {fn} in source run: {src_f}")
        shutil.copy2(src_f, out / fn)
        copied.append(fn)

    enc_sha = sha256_file(out / "encoder.pt")
    q_sha = sha256_file(out / "query_encoder.pt")

    with open(source / "summary.json", "r") as f:
        cfg = json.load(f).get("config", {})

    manifest = {
        "name": name,
        "source_run": str(source),
        "copied_files": copied,
        "encoder_sha256": enc_sha,
        "query_encoder_sha256": q_sha,
        "config": cfg,
        "frozen_metrics": _frozen_metrics(source),
        "noise_floor": compute_noise_floor(seed_dirs),
        "gate_rule": "threshold = baseline_mean + max(spec_delta, 1*std); "
                     "see lock_baseline.gate_threshold",
    }
    with open(manifest_path(out), "w") as f:
        json.dump(manifest, f, indent=2)
    _print_summary(manifest)
    return manifest


def _print_summary(m: dict) -> None:
    print()
    print("=" * 84)
    print(f"v3.1 baseline locked: {m['name']}")
    print(f"source: {m['source_run']}")
    print("=" * 84)
    print(f"  encoder.pt       sha256 {m['encoder_sha256'][:16]}...")
    print(f"  query_encoder.pt sha256 {m['query_encoder_sha256'][:16]}...")
    fv = m["frozen_metrics"].get("final_val_overall", {})
    print("\nfrozen val metrics:")
    for k in ("p@5", "ndcg@5", "p@10", "ndcg@10", "ndcg@20"):
        if k in fv:
            print(f"  {k:<10} {fv[k]:.4f}")
    ic = m["frozen_metrics"].get("intrinsic_corpus", {})
    if ic:
        print(f"  corpus edge_prec@5  {ic.get('edge_precision_at_k_mean')}")
        print(f"  corpus silhouette   {ic.get('silhouette_mean')}")
        print(f"  random edge_prec    {ic.get('random_baseline_edge_prec_mean')}")
    nf = m["noise_floor"]
    print(f"\nnoise floor over {nf['n_seeds']} seeds:")
    for k in ("ndcg@10", "ndcg@20", "intrinsic_edge_prec@5"):
        b = nf[k]
        print(f"  {k:<24} mean={b['mean']:.4f}  std={b['std']:.4f}  "
              f"values={['%.4f' % v for v in b['values']]}")
    # The headline gate, materialized once for reference.
    nd = nf["ndcg@10"]
    spec_delta = 0.52 - nd["mean"]
    print(f"\nP2 ndcg@10 gate = max(0.52, base+1std) = "
          f"{max(0.52, gate_threshold(nf, 'ndcg@10', spec_delta)):.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=str,
                   default="runs/sweep_arch_hyp/h128_l4_seed1")
    p.add_argument("--out", type=str,
                   default="runs/v3.1-baseline-hyp-h128-l4-seed1")
    p.add_argument("--name", type=str, default="v3.1-baseline-hyp-h128-l4-seed1")
    p.add_argument(
        "--seeds-family", type=str, nargs="+", default=None,
        help="Run dirs forming the noise-floor cohort. Defaults to every "
             "sibling 'h128_l4_seed*' dir of --source.",
    )
    args = p.parse_args()

    source = Path(args.source)
    if args.seeds_family:
        seed_dirs = [Path(s) for s in args.seeds_family]
    else:
        # auto-discover the seed family from the source's parent
        stem = source.name  # e.g. h128_l4_seed1
        prefix = stem.rsplit("_seed", 1)[0] + "_seed"
        seed_dirs = sorted(
            d for d in source.parent.glob(f"{prefix}*") if d.is_dir()
        )
        if not seed_dirs:
            seed_dirs = [source]

    lock_baseline(
        source=source, out=Path(args.out), name=args.name, seed_dirs=seed_dirs
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
