r"""Localhost HTTP adapter (plan T5b): thin transport over
``EvidencePipeline`` under ``/api/v1``. Single-user, localhost-only.

Status mapping (plan: Artifacts and APIs):
- 200 reused / 201 new revision on compile
- 400 malformed request
- 404 unknown packet/revision/file
- 422 unconfirmed anchors or unsatisfied packet obligations (ContractError)
- 423 workspace write lock held (with Retry-After)
- 503 Neo4j or model dependency unavailable

No endpoint accepts arbitrary Cypher; revision files are served from an
allowlist only. The interpreter endpoint is wired when T6 lands.
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request

from .contracts import ContractError
from .pipeline import DEFAULT_WORKSPACE, EvidencePipeline
from .store import WorkspaceLockedError

_REVISION_FILES = ("question.json", "candidates.json", "packet.json",
                   "packet.md", "manifest.json", "interpretation.md",
                   "interpretation_meta.json")


def create_app(workspace_root: str | Path = DEFAULT_WORKSPACE,
               pipeline: EvidencePipeline | None = None) -> Flask:
    from .store import Workspace

    app = Flask(__name__)
    state: dict = {"pipeline": pipeline, "workspace_root": workspace_root}
    workspace = Workspace(workspace_root)   # read endpoints never need Neo4j

    def pipe() -> EvidencePipeline:
        if state["pipeline"] is None:
            state["pipeline"] = EvidencePipeline(state["workspace_root"])
        return state["pipeline"]

    # -- error mapping -------------------------------------------------------

    @app.errorhandler(ContractError)
    def _contract(ex):
        return jsonify({"error": str(ex)}), 422

    @app.errorhandler(WorkspaceLockedError)
    def _locked(ex):
        return jsonify({"error": str(ex)}), 423, {"Retry-After": "5"}

    @app.errorhandler(KeyError)
    def _badkey(ex):
        return jsonify({"error": f"missing field: {ex}"}), 400

    @app.errorhandler(Exception)
    def _boom(ex):
        name = type(ex).__name__
        if "ServiceUnavailable" in name or "Neo4j" in name:
            return jsonify({"error": f"{name}: {ex}"}), 503
        raise ex

    # -- endpoints -------------------------------------------------------------

    @app.get("/api/v1/health")
    def health():
        try:
            p = pipe()
            return jsonify({"ok": True, "snapshot_epoch": p.src.snapshot_epoch})
        except Exception as ex:  # noqa: BLE001 -- health must not 500
            return jsonify({"ok": False,
                            "error": f"{type(ex).__name__}: {ex}"}), 503

    @app.post("/api/v1/anchors/resolve")
    def resolve():
        body = request.get_json(force=True)
        _, cands = pipe().resolve(body["question"])
        from .resolver import AnchorResolver
        return jsonify({
            "snapshot_epoch": pipe().src.snapshot_epoch,
            "resolution_hash": AnchorResolver.resolution_hash(cands),
            "candidates": [{
                "rank": c.rank, "key": c.public_key, "label": c.label,
                "name": c.display_name, "method": c.match_method,
                "fallback": c.key_is_fallback} for c in cands]})

    @app.post("/api/v1/packets/compile")
    def compile_():
        body = request.get_json(force=True)
        result = pipe().compile(
            body["question"], body["confirm"],
            confirmed_by=body["confirmed_by"],
            confirmed_at=body.get("confirmed_at", ""))
        return jsonify(result), (200 if result.get("reused") else 201)

    @app.post("/api/v1/packets/preview")
    def preview():
        body = request.get_json(force=True)
        result = pipe().compile(
            body["question"], body["confirm"],
            confirmed_by=body["confirmed_by"],
            confirmed_at=body.get("confirmed_at", ""), write=False)
        return jsonify(result)

    @app.get("/api/v1/packets")
    def list_packets():
        ws = workspace
        out = []
        if ws.packets.exists():
            for d in sorted(ws.packets.iterdir()):
                latest = ws.latest(d.name)
                if latest:
                    out.append(latest)
        return jsonify({"packets": out})

    @app.get("/api/v1/packets/<pid>")
    def get_packet(pid):
        ws = workspace
        latest = ws.latest(pid)
        if latest is None:
            return jsonify({"error": f"no such packet: {pid}"}), 404
        manifest = ws.read_manifest(pid, latest["revision"])
        return jsonify({"latest": latest, "manifest": manifest.to_dict()})

    @app.get("/api/v1/packets/<pid>/revisions/<int:rev>/<name>")
    def get_file(pid, rev, name):
        if name not in _REVISION_FILES:
            return jsonify({"error": "file not in the served allowlist"}), 404
        ws = workspace
        path = ws.revision_dir(pid, rev) / name
        if not path.exists():
            return jsonify({"error": "not found"}), 404
        data = path.read_bytes()
        if name.endswith(".json"):
            return app.response_class(data, mimetype="application/json")
        return app.response_class(data, mimetype="text/markdown")

    @app.post("/api/v1/packets/<pid>/validate")
    def validate(pid):
        from . import canonical
        from .compiler import validate_packet
        from .contracts import CandidateBundle

        ws = workspace
        latest = ws.latest(pid)
        if latest is None:
            return jsonify({"error": f"no such packet: {pid}"}), 404
        rev = int(request.args.get("revision", latest["revision"]))
        manifest = ws.read_manifest(pid, rev)
        problems = [f"hash mismatch: {n}"
                    for n, expect in sorted(manifest.output_hashes.items())
                    if canonical.hash_bytes(
                        ws.read_revision_file(pid, rev, n)) != expect]
        bundle = CandidateBundle.from_dict(json.loads(
            ws.read_revision_file(pid, rev, "candidates.json")))
        bundle.validate()
        validate_packet(json.loads(
            ws.read_revision_file(pid, rev, "packet.json")), bundle)
        return jsonify({"ref": f"{pid}@{rev:04d}",
                        "verdict": "PASS" if not problems else "FAIL",
                        "problems": problems})

    return app


def serve(host: str = "127.0.0.1", port: int = 8766,
          workspace_root: str | Path = DEFAULT_WORKSPACE) -> None:
    app = create_app(workspace_root)
    print(f"[evidence-api] building pipeline (one-time graph pull) ...")
    app.ensure_sync(lambda: None)()      # no-op; pipeline builds on first use
    print(f"[evidence-api] serving on http://{host}:{port}/api/v1 "
          f"(localhost only; no Cypher endpoint)")
    app.run(host=host, port=port, debug=False, use_reloader=False)
