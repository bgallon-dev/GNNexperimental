r"""Build a tier1-schema NPZ corpus from a multi-repo code-graph export
so ``train_v3`` (Stage A) can iterate it like the synthetic corpus.

One ``graph_NN_<repo>.npz`` per repo, written to ``--out``. Each NPZ has
the standard schema arrays + a single no-op task slot (Stage A ignores
``sample.query`` / ``sample.labels``). Answer edges are *kept* — Stage
A is unconditional pretraining; the eval harness ablates per-case
answer edges at scoring time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingest import build_npz


def discover_repos(corpus_root: Path, extra: list[str]) -> list[Path]:
    out: list[Path] = []
    if corpus_root.is_dir():
        out += sorted(
            p for p in corpus_root.iterdir()
            if (p / "nodes.jsonl").is_file()
        )
    for e in extra:
        p = Path(e)
        if (p / "nodes.jsonl").is_file():
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", default="../corpus_validation")
    ap.add_argument("--extra-repo", action="append", default=[])
    ap.add_argument("--out", default="src/data/corpus/code_v1")
    ap.add_argument("--max-nodes", type=int, default=0,
                    help="skip repos with more than this many nodes "
                    "(0 = no cap). Useful for fitting Stage-A on an 8 GB "
                    "consumer GPU — the giants (pydantic/fastapi) still "
                    "go in the eval set, just not Stage-A pretrain.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Clean stale graph_*.npz from prior builds so a re-run with a
    # different --max-nodes can't leak the previous corpus.
    for stale in out.glob("graph_*.npz"):
        stale.unlink()
    repos = discover_repos(Path(args.corpus_root), args.extra_repo)
    if not repos:
        raise SystemExit(
            f"no repos under {args.corpus_root} or --extra-repo "
            f"{args.extra_repo}"
        )

    manifest = []
    kept_idx = 0
    for rd in repos:
        cg = build_npz(
            rd, out / f"graph_{kept_idx:02d}_{rd.name}.npz",
            required_edge_ids=set(), with_dummy_task=True,
        )
        if args.max_nodes and cg.n_nodes > args.max_nodes:
            (out / f"graph_{kept_idx:02d}_{rd.name}.npz").unlink(missing_ok=True)
            print(f"  [skip] {rd.name}: {cg.n_nodes}n > --max-nodes "
                  f"{args.max_nodes}")
            continue
        manifest.append(dict(
            idx=kept_idx, name=rd.name, src=str(rd),
            n_nodes=cg.n_nodes, n_edges=cg.n_edges_kept,
            kinds=sorted(cg.kind_to_type_id),
            rels=sorted(cg.rel_to_type_id),
        ))
        print(f"  [{kept_idx}] {rd.name}: {cg.n_nodes}n / {cg.n_edges_kept}e")
        kept_idx += 1

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} graphs to {out} "
          f"({len(repos) - len(manifest)} skipped over --max-nodes)")


if __name__ == "__main__":
    main()
