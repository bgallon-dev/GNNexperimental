"""Package the scaling sweep into a zip suitable for Colab upload.

Includes:
  - kettle-graph-reasoner/src/  (all Python source; tests excluded)
  - kettle-graph-reasoner/runs/sweep_arch_hyp/h128_l4_seed1/ (shipped
    encoder + summary so the harness can compare against the frozen
    archival baseline if desired)
  - corpus_validation/<repo>/{nodes,edges,training_cases}.jsonl  +
    metadata.json + validation_report.json  (raw graph data; bulky
    raw_nodes / raw_edges / parse_events / symbol_table are excluded)
  - tutorstructure_patch/{nodes,edges,training_cases}.jsonl + same

Excludes venv, __pycache__, runs/* except the shipped checkpoint, raw_*
jsonls, and any local Colab-only artifacts. Resulting bundle is the
minimum to reproduce the sweep on Colab T4.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

KEEP_REPO_FILES = {
    "nodes.jsonl", "edges.jsonl", "training_cases.jsonl",
    "training_cases_v2.jsonl",
    "metadata.json", "validation_report.json",
}


def _add_tree(zf: zipfile.ZipFile, root: Path, arcroot: str,
              include_ext: tuple = (".py",),
              skip_dirs: tuple = ("__pycache__", "tests", ".venv_dml",
                                  ".pytest_cache")) -> int:
    n = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(s in p.parts for s in skip_dirs):
            continue
        if include_ext and p.suffix not in include_ext:
            continue
        rel = p.relative_to(root)
        zf.write(p, f"{arcroot}/{rel.as_posix()}")
        n += 1
    return n


def _add_repo_jsonls(zf: zipfile.ZipFile, repo_dir: Path,
                     arcroot: str) -> int:
    n = 0
    for name in KEEP_REPO_FILES:
        p = repo_dir / name
        if p.is_file():
            arc = f"{arcroot}/{repo_dir.name}/{name}" if arcroot \
                else f"{repo_dir.name}/{name}"
            zf.write(p, arc)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="..",
                    help="parent of kettle-graph-reasoner/, "
                    "corpus_validation/, tutorstructure_patch/")
    ap.add_argument("--out", default="../kgr_colab_bundle.zip")
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    kgr = ws / "kettle-graph-reasoner"
    cv = ws / "corpus_validation"
    tut = ws / "tutorstructure_patch"
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Source tree.
        n_py = _add_tree(zf, kgr / "src", "kettle-graph-reasoner/src",
                         include_ext=(".py",))
        # Notebooks (Colab launchers).
        n_nb = 0
        nb_dir = kgr / "notebooks"
        if nb_dir.is_dir():
            n_nb = _add_tree(zf, nb_dir, "kettle-graph-reasoner/notebooks",
                             include_ext=(".ipynb",))
        # Stage-A corpus is now materialized from the pack at runtime via
        # src.codegraph.pack_to_corpus — no NPZs shipped in the bundle.
        n_corpus = 0
        # Shipped baseline checkpoint (small; useful for comparison).
        ck = kgr / "runs" / "sweep_arch_hyp" / "h128_l4_seed1"
        n_ck = 0
        for name in ("encoder.pt", "query_encoder.pt", "summary.json"):
            p = ck / name
            if p.is_file():
                zf.write(p, f"kettle-graph-reasoner/runs/sweep_arch_hyp/"
                            f"h128_l4_seed1/{name}")
                n_ck += 1
        # Corpus_validation raw jsonls (per repo, only the files the
        # harness actually reads).
        n_repos = 0
        n_repo_files = 0
        if cv.is_dir():
            for repo in sorted(cv.iterdir()):
                if (repo / "nodes.jsonl").is_file():
                    n_repo_files += _add_repo_jsonls(zf, repo, "corpus_validation")
                    n_repos += 1
        # tutorstructure_patch.
        n_tut = 0
        if (tut / "nodes.jsonl").is_file():
            n_tut = _add_repo_jsonls(zf, tut, "")
            # Put tutorstructure as a sibling (matches workspace layout
            # the harness expects via --extra-repo ../tutorstructure_patch).
            # Move under its own root by rewriting the archive entry name:
            # the _add_repo_jsonls call wrote them under
            # "/tutorstructure_patch/<name>"; that's correct as a top-
            # level dir of the zip. No fixup needed.

    print(f"wrote {out}")
    print(f"  source .py files       : {n_py}")
    print(f"  notebooks              : {n_nb}")
    print(f"  code_v1 NPZ corpus     : {n_corpus}")
    print(f"  shipped ckpt files     : {n_ck}")
    print(f"  corpus_validation repos: {n_repos}  ({n_repo_files} jsonl files)")
    print(f"  tutorstructure files   : {n_tut}")
    print(f"  size                   : {out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
