r"""Immutable revision store + atomic writer + workspace lock (plan T1).

Layout (plan: Artifacts and APIs)::

    research_workspace/
    ├── annotations/*.jsonl
    └── packets/<packet_id>/
        ├── latest.json          # pointer {packet_id, revision, manifest_hash}
        ├── events.jsonl         # append-only build/reuse events
        └── revisions/0001/      # immutable once written
            question.json candidates.json packet.json packet.md manifest.json

Rules enforced here:
- Revisions are immutable: writing into an existing revision dir raises.
- Reuse: if the latest revision's manifest has the same ``dependency_hash``,
  no new revision is minted (``reused=True``).
- Atomicity: a revision is staged in a temp dir sibling and promoted with
  ``os.replace``; a crash mid-build leaves no partial revision visible.
- Single writer: an exclusive lock file guards the whole workspace;
  a held lock surfaces as ``WorkspaceLockedError`` (HTTP 423 upstream).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import canonical
from .contracts import BuildManifest, ContractError

LATEST = "latest.json"
EVENTS = "events.jsonl"
LOCK = ".lock"
REVISION_WIDTH = 4


class WorkspaceLockedError(RuntimeError):
    """Another process holds the workspace write lock."""


class ImmutabilityError(RuntimeError):
    """Attempt to modify an existing revision."""


@dataclass(frozen=True)
class RevisionRef:
    packet_id: str
    revision: int
    reused: bool
    path: Path

    @property
    def name(self) -> str:
        return f"{self.packet_id}@{self.revision:0{REVISION_WIDTH}d}"


class WorkspaceLock:
    """Exclusive create of a lock file; content is the holder's pid."""

    def __init__(self, root: Path):
        self._path = root / LOCK
        self._fd: int | None = None

    def __enter__(self) -> "WorkspaceLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = ""
            try:
                holder = self._path.read_text("utf-8").strip()
            except OSError:
                pass
            raise WorkspaceLockedError(
                f"workspace is locked by pid {holder or 'unknown'} "
                f"({self._path}); retry after it finishes") from None
        os.write(self._fd, str(os.getpid()).encode("ascii"))
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self._path.unlink()
        except OSError:
            pass


class Workspace:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.annotations = self.root / "annotations"
        self.packets = self.root / "packets"

    def lock(self) -> WorkspaceLock:
        return WorkspaceLock(self.root)

    def packet_dir(self, packet_id: str) -> Path:
        return self.packets / packet_id

    def revision_dir(self, packet_id: str, revision: int) -> Path:
        return (self.packet_dir(packet_id) / "revisions"
                / f"{revision:0{REVISION_WIDTH}d}")

    # -- reading ---------------------------------------------------------

    def latest(self, packet_id: str) -> dict | None:
        p = self.packet_dir(packet_id) / LATEST
        if not p.exists():
            return None
        return json.loads(p.read_text("utf-8"))

    def read_manifest(self, packet_id: str, revision: int) -> BuildManifest:
        p = self.revision_dir(packet_id, revision) / "manifest.json"
        return BuildManifest.from_dict(json.loads(p.read_text("utf-8")))

    def read_revision_file(self, packet_id: str, revision: int,
                           filename: str) -> bytes:
        return (self.revision_dir(packet_id, revision) / filename).read_bytes()

    # -- writing ---------------------------------------------------------

    def write_revision(self, manifest: BuildManifest,
                       files: dict[str, bytes]) -> RevisionRef:
        """Persist one immutable revision atomically.

        ``files`` maps filename -> exact bytes (already canonical).
        ``manifest.json`` is derived from ``manifest`` here -- callers never
        hand-serialize it -- with ``output_hashes`` filled from ``files``.
        Returns a ref; ``reused=True`` (and no write) when the latest
        revision has the same dependency hash.
        """
        manifest.validate()
        if "manifest.json" in files:
            raise ContractError("manifest.json is derived; do not pass it in")
        packet_id = manifest.question_id
        latest = self.latest(packet_id)
        if latest is not None:
            prev = self.read_manifest(packet_id, latest["revision"])
            if prev.dependency_hash == manifest.dependency_hash:
                return RevisionRef(packet_id, latest["revision"], True,
                                   self.revision_dir(packet_id,
                                                     latest["revision"]))
        revision = 1 if latest is None else latest["revision"] + 1
        final = self.revision_dir(packet_id, revision)
        if final.exists():
            raise ImmutabilityError(f"revision already exists: {final}")

        output_hashes = {name: canonical.hash_bytes(data)
                         for name, data in sorted(files.items())}
        stamped = BuildManifest.from_dict(
            {**manifest.to_dict(), "revision": revision,
             "output_hashes": output_hashes})
        manifest_bytes = canonical.canonical_bytes(stamped.to_dict())

        final.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".stage-{revision:04d}-",
                                        dir=final.parent))
        try:
            for name, data in sorted(files.items()):
                (staging / name).write_bytes(data)
            (staging / "manifest.json").write_bytes(manifest_bytes)
            os.replace(staging, final)          # atomic promote
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        pointer = {"packet_id": packet_id, "revision": revision,
                   "manifest_hash": canonical.hash_bytes(manifest_bytes)}
        self._write_atomic(self.packet_dir(packet_id) / LATEST,
                           canonical.canonical_bytes(pointer))
        self.append_event(packet_id, {"event": "revision_created",
                                      "revision": revision,
                                      "dependency_hash":
                                          stamped.dependency_hash})
        return RevisionRef(packet_id, revision, False, final)

    def append_event(self, packet_id: str, event: dict) -> None:
        p = self.packet_dir(packet_id) / EVENTS
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.write(canonical.canonical_dumps(event) + "\n")

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
