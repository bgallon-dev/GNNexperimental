r"""Materialize per-repo NPZs from a kgr_pack v0.2 binary corpus.

Stage-A (``src/training/train_v3.py``) trains via
``src.data.corpus_dataset.CorpusDataset`` which globs ``graph_*.npz`` from
a directory and expects each NPZ to carry the tier-1 schema arrays plus
at least one task slot. Pack format is mmap-friendly and ~45× smaller on
disk, but ``CorpusDataset`` doesn't speak it. This script bridges:

  pack/ ──(iter_nodes / iter_edges via pack_loader)─→ NPZ dir/

Restricts the emit list to the **train repos** in a stratified split
file so the encoder never sees the held-out test repos during Stage-A
pretraining.

Usage:

  python -m src.codegraph.pack_to_corpus \\
      --pack ../packed_training/real.debug \\
      --split src/codegraph/splits/stratified_80_20.json \\
      --out /content/stage_a_corpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingest import build_npz
from .pack_loader import PackContext


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True,
                    help="Pack directory (must contain header.bin)")
    ap.add_argument("--split", default="",
                    help="Optional split JSON; if given, only emit "
                    "NPZs for repos in split['train']. Test repos are "
                    "skipped to prevent Stage-A leakage.")
    ap.add_argument("--out", required=True,
                    help="Output directory for graph_NN_<repo>.npz files")
    ap.add_argument("--max-nodes", type=int, default=0,
                    help="Skip repos with more than N nodes. Stage-A on "
                    "graphs >~100k nodes OOMs at h>=256 on A100 (~40GB). "
                    "Recommended: 80000 to keep 12/17 repos while excluding "
                    "pandas/django/scipy/scikit-learn/numpy.")
    args = ap.parse_args()

    pack_dir = Path(args.pack)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_only = None
    if args.split:
        split = json.loads(Path(args.split).read_text())
        train_only = set(split["train"])
        print(f"  split: train={len(train_only)} repos "
              f"({len(split['test'])} test repos excluded)")

    with PackContext(pack_dir) as ctx:
        summaries = list(ctx.iter_repos())
        by_idx = {r.idx: r for r in summaries}
        repos = sorted([(r.idx, r.name) for r in summaries], key=lambda x: x[1])
        if train_only is not None:
            repos = [(i, n) for i, n in repos if n in train_only]
        if args.max_nodes > 0:
            keep: list[tuple[int, str]] = []
            for repo_idx, name in repos:
                n = by_idx[repo_idx].n_nodes
                if n > args.max_nodes:
                    print(f"  [skip giant] {name}: {n:,} nodes "
                          f"> --max-nodes {args.max_nodes:,}")
                else:
                    keep.append((repo_idx, name))
            repos = keep
        print(f"  emitting {len(repos)} NPZs to {out_dir}")
        emit_i = 0
        for repo_idx, name in repos:
            out_npz = out_dir / f"graph_{emit_i:02d}_{name}.npz"
            emit_i += 1
            if out_npz.exists():
                print(f"    [skip] {out_npz.name} exists")
                continue
            try:
                g = build_npz(
                    data_dir=pack_dir,  # unused when iters supplied
                    out_npz=out_npz,
                    nodes_iter=ctx.iter_nodes(repo_idx),
                    edges_iter=ctx.iter_edges(repo_idx),
                    with_dummy_task=True,  # CorpusDataset needs >=1 task slot
                )
                print(f"    wrote {out_npz.name}  "
                      f"({g.n_nodes}n / {g.n_edges_kept}e)")
            except Exception as e:
                print(f"    [FAIL] {name}: {e!r}")
                raise

    # Minimal manifest (mirrors code_v1 layout).
    manifest = {
        "n_graphs": len(list(out_dir.glob("graph_*.npz"))),
        "source_pack": str(pack_dir),
        "split_train_only": args.split or None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
