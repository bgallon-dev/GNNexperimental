"""CLI entrypoint for the evidence workspace (plan: Artifacts and APIs).

All plan surfaces are wired: ``anchors resolve``, ``packet preview /
compile / validate / diff / interpret``, ``workbench generate / ingest``
(T2 support) and ``serve`` (localhost HTTP adapter). The CLI and the API
share one implementation (``pipeline.py``).

Anchor confirmation is a HUMAN act: ``packet compile`` requires
``--confirm key1,key2 --confirmed-by NAME`` and the keys must come from a
fresh resolver run (the confirmation is hash-bound to that candidate
list; a stale confirmation fails).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLAN = "Docs/EVIDENCE_WORKSPACE_PLAN.md"
_ROOT = Path(__file__).resolve().parents[2]


def _pipeline(args: argparse.Namespace):
    from .pipeline import EvidencePipeline

    return EvidencePipeline(getattr(args, "workspace", None)
                            or _ROOT / "research_workspace")


# -- handlers ------------------------------------------------------------------

def cmd_anchors_resolve(args: argparse.Namespace) -> int:
    from .resolver import AnchorResolver

    pipe = _pipeline(args)
    try:
        _, cands = pipe.resolve(args.question)
    finally:
        pipe.close()
    print(json.dumps({
        "snapshot_epoch": pipe.src.snapshot_epoch,
        "resolution_hash": AnchorResolver.resolution_hash(cands),
        "candidates": [{
            "rank": c.rank, "key": c.public_key, "label": c.label,
            "name": c.display_name, "method": c.match_method,
            "fallback": c.key_is_fallback,
        } for c in cands],
    }, indent=2, sort_keys=True))
    return 0


def _compile(args: argparse.Namespace, write: bool) -> int:
    pipe = _pipeline(args)
    try:
        result = pipe.compile(
            args.question,
            [k.strip() for k in args.confirm.split(",") if k.strip()],
            confirmed_by=args.confirmed_by, confirmed_at=args.confirmed_at,
            write=write)
    finally:
        pipe.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_packet_preview(args: argparse.Namespace) -> int:
    return _compile(args, write=False)


def cmd_packet_compile(args: argparse.Namespace) -> int:
    return _compile(args, write=True)


def cmd_packet_compile_bundle(args: argparse.Namespace) -> int:
    """Offline compile from persisted question.json + candidates.json.

    No database, no pipeline: loads the two artifacts the online compile
    already writes, ranks with an offline lane (or a supplied precomputed
    ranking), and emits the canonical packet bytes on stdout. The whole
    point is byte-identity with the in-process compile (see
    tests/test_evidence_offline.py)."""
    from . import canonical
    from .compiler import render_markdown
    from .contracts import ContractError
    from .offline import (
        compile_bundle,
        load_annotations,
        load_bundle,
        load_question,
        load_ranking,
    )

    # Load + compile share one guard: malformed input files (bad JSON, a BOM
    # already tolerated by the loaders, missing fields), a stale/mismatched
    # supplied ranking, and compile-time contract failures all report cleanly
    # on stderr with exit 2 rather than a raw traceback.
    try:
        question = load_question(args.question)
        bundle = load_bundle(args.candidates)
        ranking = load_ranking(args.ranking) if args.ranking else None
        annotations = (load_annotations(args.annotations)
                       if args.annotations else ())
        packet = compile_bundle(
            question, bundle, strategy=args.strategy, ranking=ranking,
            annotations=annotations,
            allow_partial_ranking=args.allow_partial_ranking)
    except (ContractError, ValueError, KeyError, OSError) as ex:
        print(json.dumps({"error": type(ex).__name__, "detail": str(ex)},
                         indent=2, sort_keys=True), file=sys.stderr)
        return 2

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "question.json").write_bytes(
            canonical.canonical_bytes(question.to_dict()))
        (out / "candidates.json").write_bytes(
            canonical.canonical_bytes(bundle.to_dict()))
        (out / "packet.json").write_bytes(canonical.canonical_bytes(packet))
        (out / "packet.md").write_bytes(render_markdown(packet).encode("utf-8"))
        print(json.dumps({
            "written": True, "out": str(out),
            "question_id": question.question_id,
            "strategy": "supplied" if ranking is not None else args.strategy,
            "core_evidence": len(packet["core_evidence"]),
        }, indent=2, sort_keys=True), file=sys.stderr)

    # canonical packet bytes on stdout (already newline-terminated); pipe-friendly
    sys.stdout.buffer.write(canonical.canonical_bytes(packet))
    return 0


def _parse_ref(ref: str) -> tuple[str, int | None]:
    if "@" in ref:
        pid, _, rev = ref.partition("@")
        return pid, int(rev)
    return ref, None


def _workspace(args: argparse.Namespace):
    from .store import Workspace

    return Workspace(args.workspace)


def cmd_packet_validate(args: argparse.Namespace) -> int:
    from . import canonical
    from .compiler import validate_packet
    from .contracts import CandidateBundle

    ws = _workspace(args)
    pid, rev = _parse_ref(args.ref)
    if rev is None:
        latest = ws.latest(pid)
        if latest is None:
            print(f"no such packet: {pid}", file=sys.stderr)
            return 1
        rev = latest["revision"]
    manifest = ws.read_manifest(pid, rev)
    problems = []
    for name, expect in sorted(manifest.output_hashes.items()):
        got = canonical.hash_bytes(ws.read_revision_file(pid, rev, name))
        if got != expect:
            problems.append(f"hash mismatch: {name}")
    bundle = CandidateBundle.from_dict(json.loads(
        ws.read_revision_file(pid, rev, "candidates.json")))
    bundle.validate()
    packet = json.loads(ws.read_revision_file(pid, rev, "packet.json"))
    validate_packet(packet, bundle)
    verdict = "PASS" if not problems else "FAIL"
    print(json.dumps({"ref": f"{pid}@{rev:04d}", "verdict": verdict,
                      "problems": problems}, indent=2, sort_keys=True))
    return 0 if not problems else 1


def cmd_packet_diff(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    manifests = []
    for ref in (args.old_ref, args.new_ref):
        pid, rev = _parse_ref(ref)
        if rev is None:
            rev = ws.latest(pid)["revision"]
        manifests.append((f"{pid}@{rev:04d}",
                          ws.read_manifest(pid, rev).to_dict()))
    (na, ma), (nb, mb) = manifests
    changed = {k: {"old": ma[k], "new": mb[k]}
               for k in sorted(set(ma) | set(mb))
               if ma.get(k) != mb.get(k)}
    print(json.dumps({"old": na, "new": nb, "changed_fields": changed},
                     indent=2, sort_keys=True))
    return 0


def cmd_packet_interpret(args: argparse.Namespace) -> int:
    from .interpret import interpret_packet

    ws = _workspace(args)
    pid, rev = _parse_ref(args.ref)
    result = interpret_packet(ws, pid, revision=rev,
                              model=args.model or None)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in ("added", "exists") else 1


def cmd_workbench_generate(args: argparse.Namespace) -> int:
    from .neo4j_backend import _read_session
    from .workbench import generate_worksheet

    pipe = _pipeline(args)
    out = args.out
    if not out:
        p = Path(args.question)
        out = str(p.parent.parent / "worksheets"
                  / f"{p.parent.name}_{p.stem}.csv")
    try:
        summary = generate_worksheet(
            args.question, pipe.resolver,
            lambda: _read_session(pipe.src._drv), out)
    finally:
        pipe.close()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_workbench_ingest(args: argparse.Namespace) -> int:
    from .neo4j_backend import _read_session
    from .resolver import DEFAULT_KEY_PROPERTIES
    from .workbench import ingest_worksheet

    pipe = _pipeline(args)
    try:
        summary = ingest_worksheet(
            args.question, args.worksheet,
            lambda: _read_session(pipe.src._drv), DEFAULT_KEY_PROPERTIES,
            anchor_keys=[k.strip() for k in args.anchors.split(",")
                         if k.strip()],
            labeled_by=args.labeled_by, labeled_at=args.labeled_at)
    finally:
        pipe.close()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .api import serve

    serve(host=args.host, port=args.port, workspace_root=args.workspace)
    return 0


# -- parser --------------------------------------------------------------------

def _add_workspace(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("--workspace",
                     default=str(_ROOT / "research_workspace"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evidence",
        description="Deterministic evidence-packet engine (see %s)" % PLAN)
    sub = parser.add_subparsers(dest="command", required=True)

    anchors = sub.add_parser("anchors", help="anchor resolution")
    anchors_sub = anchors.add_subparsers(dest="subcommand", required=True)
    resolve = anchors_sub.add_parser(
        "resolve", help="suggest anchor candidates for a question fixture")
    resolve.add_argument("--question", required=True,
                         help="fixture/question JSON path")
    _add_workspace(resolve)
    resolve.set_defaults(func=cmd_anchors_resolve)

    packet = sub.add_parser("packet", help="evidence packet operations")
    packet_sub = packet.add_subparsers(dest="subcommand", required=True)

    for name, func in (("preview", cmd_packet_preview),
                       ("compile", cmd_packet_compile)):
        cmd = packet_sub.add_parser(name)
        cmd.add_argument("--question", required=True)
        cmd.add_argument("--confirm", required=True,
                         help="comma-separated confirmed public keys "
                              "(must appear in a fresh resolver run)")
        cmd.add_argument("--confirmed-by", required=True)
        cmd.add_argument("--confirmed-at", default="",
                         help="ISO date of the confirmation act")
        _add_workspace(cmd)
        cmd.set_defaults(func=func)

    cb = packet_sub.add_parser(
        "compile-bundle",
        help="offline packet compile from a persisted question.json + "
             "candidates.json (no database). bfs/lexical lanes compute "
             "offline; kgr must be supplied precomputed via --ranking.")
    cb.add_argument("--question", required=True,
                    help="serialized ResearchQuestion JSON "
                         "(the online pipeline's question.json)")
    cb.add_argument("--candidates", required=True,
                    help="serialized CandidateBundle JSON (candidates.json)")
    cb.add_argument("--strategy", default="bfs", metavar="bfs|lexical",
                    help="offline ranking lane (default bfs); kgr is gated -- "
                         "supply it via --ranking instead")
    cb.add_argument("--ranking", default="",
                    help="optional precomputed RankedCandidate list JSON; "
                         "validated against the bundle then used verbatim, "
                         "enabling offline compile of an online-computed kgr "
                         "ranking")
    cb.add_argument("--allow-partial-ranking", action="store_true",
                    help="permit a supplied --ranking that does not cover "
                         "every domain node (e.g. a kgr ranking that dropped "
                         "unembedded nodes); off by default so a stale ranking "
                         "with a mismatched node set is rejected")
    cb.add_argument("--annotations", default="",
                    help="optional AnnotationRecord list JSON (sidecar)")
    cb.add_argument("--out", default="",
                    help="optional dir to also write question/candidates/"
                         "packet.json/packet.md as canonical bytes")
    cb.set_defaults(func=cmd_packet_compile_bundle)

    validate = packet_sub.add_parser("validate")
    validate.add_argument("ref", help="<packet_id>[@revision]")
    _add_workspace(validate)
    validate.set_defaults(func=cmd_packet_validate)

    diff = packet_sub.add_parser("diff")
    diff.add_argument("old_ref")
    diff.add_argument("new_ref")
    _add_workspace(diff)
    diff.set_defaults(func=cmd_packet_diff)

    interp = packet_sub.add_parser(
        "interpret", help="optional packet-only LLM memo (T6; requires a "
                          "local LLM server)")
    interp.add_argument("ref", help="<packet_id>[@revision]")
    interp.add_argument("--model", default="")
    _add_workspace(interp)
    interp.set_defaults(func=cmd_packet_interpret)

    wb = sub.add_parser("workbench", help="labeling workbench (T2 support)")
    wb_sub = wb.add_subparsers(dest="subcommand", required=True)
    gen = wb_sub.add_parser(
        "generate", help="build a seeded-random grading worksheet CSV")
    gen.add_argument("--question", required=True)
    gen.add_argument("--out", default="")
    _add_workspace(gen)
    gen.set_defaults(func=cmd_workbench_generate)
    ing = wb_sub.add_parser(
        "ingest", help="fold a graded worksheet back into the fixture")
    ing.add_argument("--question", required=True)
    ing.add_argument("--worksheet", required=True)
    ing.add_argument("--anchors", required=True,
                     help="comma-separated confirmed anchor public keys "
                          "(hop locality is computed against these)")
    ing.add_argument("--labeled-by", required=True)
    ing.add_argument("--labeled-at", required=True)
    _add_workspace(ing)
    ing.set_defaults(func=cmd_workbench_ingest)

    serve = sub.add_parser("serve", help="run the localhost HTTP adapter")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    _add_workspace(serve)
    serve.set_defaults(func=cmd_serve)

    return parser


def _ensure_hashseed_for_module_run() -> None:
    """Pin PYTHONHASHSEED=0 preserving the ``-m src.evidence`` invocation.

    ``src.service.determinism.ensure_pythonhashseed`` re-execs with
    ``sys.argv`` verbatim, which degrades a ``-m`` run into a plain-script
    run of ``__main__.py`` (no package context, relative imports break).
    This wrapper re-execs with the module form instead; once the seed is
    pinned the downstream call in ``canonical.deterministic_runtime`` is
    a no-op."""
    import os
    import subprocess

    if os.environ.get("PYTHONHASHSEED") == "0":
        return
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    raise SystemExit(subprocess.run(
        [sys.executable, "-m", "src.evidence", *sys.argv[1:]],
        env=env, cwd=str(_ROOT)).returncode)


def main(argv: list[str] | None = None) -> int:
    _ensure_hashseed_for_module_run()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
