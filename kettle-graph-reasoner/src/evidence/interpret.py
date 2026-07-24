r"""Optional packet-only interpreter (plan T6).

Boundaries (non-negotiable):
- The interpreter receives ONLY ``packet.json`` -- no graph, no database,
  no resolver. It cannot modify packet facts.
- Every substantive paragraph must cite packet evidence IDs (``[E03]``);
  citing an unknown ID, or writing substantive paragraphs without
  citations, QUARANTINES the output -- it is stored for inspection but
  never linked from the revision.
- Model, prompt and response hashes are recorded.

Uses the serving layer's local-LLM client (``src.service.llm``: LM Studio
/ Ollama autodiscovery). No local LLM -> a clean "unavailable" result,
never a fake interpretation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import canonical
from .store import Workspace

_CITE_RE = re.compile(r"\[(E\d{2})\]")
_SUBSTANTIVE_CHARS = 120

_SYSTEM = (
    "You are a careful research historian writing a short evidence memo. "
    "You may use ONLY the evidence packet provided by the user -- no outside "
    "knowledge, no speculation beyond what the evidence supports. Cite "
    "evidence IDs in square brackets, e.g. [E03], in every substantive "
    "paragraph. Where the packet lists uncertainties, name them as open "
    "questions rather than filling gaps. Keep it under 600 words.")


def _prompt_view(packet: dict[str, Any]) -> dict[str, Any]:
    """The packet subset the model sees (keeps prompts small and audit
    simple): question, evidence with sources, uncertainties, frontier."""
    return {
        "question": packet["question"],
        "family": packet["family"],
        "core_evidence": [{
            "id": c["evidence_id"], "text": c["display"],
            "about": c["about_display"], "source": c["source"],
        } for c in packet["core_evidence"]],
        "uncertainties": [{"key": u["public_key"], "detail": u["detail"]}
                          for u in packet["uncertainties"]],
        "research_frontier": [f["kind"] for f in packet["research_frontier"]],
        "coverage_status": packet["coverage_status"],
    }


def validate_citations(text: str, valid_ids: set[str]) -> list[str]:
    problems = []
    cited = set(_CITE_RE.findall(text))
    unknown = cited - valid_ids
    if unknown:
        problems.append(f"cites unknown evidence ids: {sorted(unknown)}")
    if not cited:
        problems.append("no evidence citations at all")
    for i, para in enumerate(p.strip() for p in text.split("\n\n")):
        if len(para) >= _SUBSTANTIVE_CHARS and not _CITE_RE.search(para):
            problems.append(f"substantive paragraph {i + 1} has no citation")
    return problems


def interpret_packet(workspace: Workspace, packet_id: str,
                     revision: int | None = None,
                     model: str | None = None) -> dict[str, Any]:
    from src.service import llm

    latest = workspace.latest(packet_id)
    if latest is None:
        raise FileNotFoundError(f"no such packet: {packet_id}")
    rev = revision if revision is not None else latest["revision"]
    rev_dir = workspace.revision_dir(packet_id, rev)
    if (rev_dir / "interpretation.md").exists():
        return {"ref": f"{packet_id}@{rev:04d}", "status": "exists",
                "path": str(rev_dir / "interpretation.md")}

    if not llm.available():
        return {"ref": f"{packet_id}@{rev:04d}", "status": "unavailable",
                "detail": "no local LLM server reachable (LM Studio/Ollama "
                          "/ $KGR_LLM_BASE); interpretation skipped"}

    packet = json.loads(workspace.read_revision_file(
        packet_id, rev, "packet.json"))
    valid_ids = {c["evidence_id"] for c in packet["core_evidence"]}
    user_prompt = canonical.canonical_dumps(_prompt_view(packet))
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_prompt}]
    reply = llm.chat(messages, model=model)
    # llm.chat returns {'text', 'model', 'usage'} (explorer contract);
    # tolerate a bare string for other clients
    if isinstance(reply, dict):
        text = str(reply.get("text", ""))
        used_model = str(reply.get("model") or model
                         or llm.discover().get("default"))
    else:
        text = str(reply)
        used_model = model or llm.discover().get("default")

    problems = validate_citations(text, valid_ids)
    meta = {
        "schema_version": "0.1.0",
        "model": used_model,
        "prompt_sha256": canonical.hash_bytes(user_prompt.encode("utf-8")),
        "response_sha256": canonical.hash_bytes(text.encode("utf-8")),
        "citation_validation": ("pass" if not problems else "fail"),
        "problems": problems,
        "valid_evidence_ids": sorted(valid_ids),
    }
    if problems:
        qdir = workspace.packet_dir(packet_id) / "quarantine"
        qdir.mkdir(parents=True, exist_ok=True)
        (qdir / f"{rev:04d}-interpretation.md").write_text(
            text, encoding="utf-8", newline="\n")
        (qdir / f"{rev:04d}-interpretation_meta.json").write_bytes(
            canonical.canonical_bytes(meta))
        workspace.append_event(packet_id, {
            "event": "interpretation_quarantined", "revision": rev,
            "problems": problems})
        return {"ref": f"{packet_id}@{rev:04d}", "status": "quarantined",
                "problems": problems, "path": str(qdir)}

    # additive-once write into the revision (never overwrite; the
    # deterministic files and their manifest hashes are untouched)
    (rev_dir / "interpretation.md").write_text(
        text, encoding="utf-8", newline="\n")
    (rev_dir / "interpretation_meta.json").write_bytes(
        canonical.canonical_bytes(meta))
    workspace.append_event(packet_id, {
        "event": "interpretation_added", "revision": rev,
        "response_sha256": meta["response_sha256"]})
    return {"ref": f"{packet_id}@{rev:04d}", "status": "added",
            "model": meta["model"],
            "path": str(rev_dir / "interpretation.md")}
