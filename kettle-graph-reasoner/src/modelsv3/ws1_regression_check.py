r"""v3.1.x WS1 — bit-exactness regression check.

The WS1 distance-ops refactor MUST NOT move any number the v3.1 /
scaling results were produced with (all at N <= the chunked cap). This
script re-diffs the three rewired baseline evals against the FROZEN
`*_baseline.json` (produced by the pre-refactor code) and asserts every
numeric leaf is identical within 1e-9 (NaN==NaN allowed). It walks the
*entire* JSON tree, not a curated subset, so there is no room for
silent drift; non-numeric leaves (paths) are skipped.

This is the WS1 safety net (same discipline as v3.1's frozen-encoder
assertion). ASCII-only output; non-zero exit on any mismatch.

Usage
-----
    # 1. re-run the three evals with the refactored code:
    py -m src.modelsv3.eval_candidate_recall      --checkpoint <run>/encoder.pt --task 2 --out <new>/candidate_recall.json
    py -m src.modelsv3.eval_retrieval_nn_filtered --checkpoint <run>/encoder.pt --task 2 --out <new>/retrieval_nn_filtered.json
    py -m src.modelsv3.eval_retrieval_midpoint    --checkpoint <run>/encoder.pt --task 2 --out <new>/retrieval_midpoint.json
    # 2. assert no drift:
    py -m src.modelsv3.ws1_regression_check --run runs/v3.1-baseline-hyp-h128-l4-seed1 --new-dir runs/_ws1check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

TOL = 1e-9

# new-file name -> frozen baseline name (in --run)
PAIRS = {
    "candidate_recall.json": "candidate_recall_baseline.json",
    "retrieval_nn_filtered.json": "retrieval_nn_filtered_baseline.json",
    "retrieval_midpoint.json": "retrieval_midpoint_baseline.json",
}


def _diff(a, b, path: str, out: list[str]) -> None:
    """Recursively compare two JSON trees; record numeric-leaf
    mismatches (>= TOL) into ``out``. Strings/paths are ignored."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                # structural change (key added/removed) — that IS drift
                out.append(f"{path}.{k}: present in only one side")
                continue
            _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: list length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            out.append(f"{path}: bool {a} != {b}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        an, bn = math.isnan(fa), math.isnan(fb)
        if an and bn:
            return
        if an != bn:
            out.append(f"{path}: NaN mismatch ({a} vs {b})")
        elif abs(fa - fb) >= TOL:
            out.append(f"{path}: {fa!r} vs {fb!r}  |d|={abs(fa - fb):.3e}")
    # non-numeric, non-container leaves (str paths) -> intentionally skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str,
                    default="runs/v3.1-baseline-hyp-h128-l4-seed1",
                    help="dir holding the frozen *_baseline.json")
    ap.add_argument("--new-dir", type=str, default="runs/_ws1check",
                    help="dir holding the freshly re-run eval JSONs")
    args = ap.parse_args()

    run = Path(args.run)
    new = Path(args.new_dir)
    any_fail = False
    print("=" * 78)
    print("WS1 bit-exactness regression check (tol = 1e-9, full-tree diff)")
    print(f"baseline: {run}")
    print(f"new:      {new}")
    print("=" * 78)

    for new_name, base_name in PAIRS.items():
        np_ = new / new_name
        bp = run / base_name
        if not bp.exists():
            print(f"  [SKIP] {new_name}: no baseline {bp.name}")
            continue
        if not np_.exists():
            print(f"  [FAIL] {new_name}: not re-run (missing {np_})")
            any_fail = True
            continue
        a = json.loads(np_.read_text())
        b = json.loads(bp.read_text())
        problems: list[str] = []
        _diff(a, b, new_name, problems)
        if not problems:
            print(f"  [PASS] {new_name}: bit-identical to baseline (<1e-9)")
        else:
            any_fail = True
            print(f"  [FAIL] {new_name}: {len(problems)} mismatch(es)")
            for p in problems[:20]:
                print(f"         {p}")
            if len(problems) > 20:
                print(f"         ... +{len(problems) - 20} more")

    print("=" * 78)
    if any_fail:
        print("RESULT: FAIL — WS1 is NOT bit-exact. Per the decision tree: "
              "check topk tie order / keepdim / triu concat order; do not "
              "ship until <1e-9.")
        return 2
    print("RESULT: PASS — WS1 refactor is bit-identical at validated sizes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
