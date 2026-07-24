r"""Generate a category-stratified train/test split JSON.

Rule (per locked design decision): from each category that has >= 2
repos in the corpus, deterministically hold out 1 repo as test. Single-
repo categories stay entirely in train. With 60 repos / 37 categories
that yields ~13 test repos covering 13 distinct categories, ~47 train
repos covering all 37 categories.

Repo discovery supports two corpus shapes:

* **jsonl directory** (default) — ``<corpus_root>/<name>/nodes.jsonl``
  per repo. Probe by checking for ``nodes.jsonl``.
* **pack directory** — a single kgr_pack v0.2 directory containing
  ``header.bin`` + ``repos.bin``. Probe by checking for ``header.bin``;
  if present, read repo names from the pack instead of the filesystem.

Names not present in the corpus are silently dropped (warned).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .category_map import CATEGORY_MAP


def _discover_repo_names(corpus_root: Path) -> list[str]:
    """Return the list of repo names present in the corpus, regardless
    of whether ``corpus_root`` is a jsonl directory or a pack directory."""
    if (corpus_root / "header.bin").is_file():
        # Pack mode — open via pack_loader and read repo names.
        from .pack_loader import PackContext
        with PackContext(corpus_root) as ctx:
            return [r.name for r in ctx.iter_repos()]
    # jsonl mode — directory of per-repo subdirs.
    return [
        p.name for p in corpus_root.iterdir()
        if p.is_dir() and (p / "nodes.jsonl").is_file()
    ]


def build_split(
    corpus_root: Path,
    seed: int = 0,
    min_repos_for_holdout: int = 2,
) -> dict:
    """Returns ``{"train": [names], "test": [names], "by_category": {...}}``.

    Test holdout selection is deterministic in ``seed`` so re-runs and
    multi-seed sweeps reproduce the same split."""
    present = set(_discover_repo_names(corpus_root))
    available = [name for name in CATEGORY_MAP if name in present]
    # Also include repos present in corpus but absent from CATEGORY_MAP —
    # treat each as its own singleton category so they end up in train.
    extra_in_corpus = sorted(present - set(CATEGORY_MAP))
    missing = [n for n in CATEGORY_MAP if n not in available]
    by_cat: dict[str, list[str]] = defaultdict(list)
    for name in available:
        by_cat[CATEGORY_MAP[name]].append(name)

    rng = random.Random(seed)
    train: list[str] = []
    test: list[str] = []
    cat_report: dict[str, dict] = {}
    for cat in sorted(by_cat):
        repos = sorted(by_cat[cat])
        if len(repos) >= min_repos_for_holdout:
            picked = rng.choice(repos)
            test.append(picked)
            train.extend(r for r in repos if r != picked)
            cat_report[cat] = {"train": [r for r in repos if r != picked],
                               "test": [picked]}
        else:
            train.extend(repos)
            cat_report[cat] = {"train": repos, "test": []}

    # Repos present in the corpus but absent from CATEGORY_MAP land in
    # train under a synthetic "uncategorized" bucket so they don't get
    # silently dropped (matches the 21-repo run pattern where attrs/click
    # /requests were singleton categories in the map).
    if extra_in_corpus:
        train.extend(extra_in_corpus)
        cat_report["__uncategorized__"] = {
            "train": extra_in_corpus, "test": [],
        }

    return {
        "seed": seed,
        "train": sorted(train),
        "test": sorted(test),
        "by_category": cat_report,
        "missing_from_corpus": sorted(missing),
        "extra_in_corpus": sorted(extra_in_corpus),
        "n_train": len(train),
        "n_test": len(test),
        "n_categories_in_test": sum(1 for v in cat_report.values() if v["test"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", default="../corpus_validation")
    ap.add_argument("--out", default="src/codegraph/splits/stratified_80_20.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    split = build_split(Path(args.corpus_root), seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2))

    print(f"wrote {out}")
    print(f"  train: {split['n_train']} repos")
    print(f"  test : {split['n_test']} repos across "
          f"{split['n_categories_in_test']} categories")
    if split["missing_from_corpus"]:
        print(f"  WARN: {len(split['missing_from_corpus'])} repos in the "
              f"category map are missing from {args.corpus_root}: "
              f"{split['missing_from_corpus'][:6]}{'...' if len(split['missing_from_corpus']) > 6 else ''}")
    print("\n  category-stratified test holdouts:")
    for cat, v in sorted(split["by_category"].items()):
        if v["test"]:
            print(f"    {cat:>26}  -> {v['test'][0]}")


if __name__ == "__main__":
    main()
