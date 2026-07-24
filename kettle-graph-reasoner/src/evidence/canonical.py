r"""Canonical serialization + determinism primitives (plan: Determinism policy).

Everything a packet revision persists flows through this module so that
byte-identity is a property of the data, not of dict insertion order,
float repr, or platform locale:

- ``canonical_dumps``  -- UTF-8 JSON, sorted keys, compact separators,
  ``allow_nan=False`` (NaN/Inf are contract violations, not data).
- micro-scores -- ranking scores become integers via
  ``round(score * 1_000_000)`` before any sort or serialization; raw
  floats are never persisted. Display strings are built from the integer
  (exact decimal, always 6 places), never through float formatting.
- ``content_hash`` / ``question_id`` -- SHA-256 over canonical bytes.
- ``deterministic_runtime`` -- pins the compile-path runtime and returns
  the settings dict recorded verbatim in ``BuildManifest``. Torch setup
  is applied lazily (only when a ranking strategy actually loads torch).

Byte-identity is guaranteed only under the same pinned environment and
manifest -- this module makes serialization exact; it cannot make two
different BLAS builds agree.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

MICRO = 1_000_000
QUESTION_ID_HEX_LEN = 20


class CanonicalizationError(ValueError):
    """Raised when a value cannot be canonically serialized."""


def canonical_dumps(obj: Any) -> str:
    """Canonical JSON text: sorted keys, compact, no NaN, non-ASCII kept."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except ValueError as ex:  # NaN/Infinity or non-serializable value
        raise CanonicalizationError(str(ex)) from ex


def canonical_bytes(obj: Any) -> bytes:
    """Canonical UTF-8 bytes with a trailing newline (POSIX-friendly files)."""
    return (canonical_dumps(obj) + "\n").encode("utf-8")


def content_hash(obj: Any) -> str:
    """SHA-256 hex digest of the canonical bytes of ``obj``."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def question_id(text: str, scope: Mapping[str, Any]) -> str:
    """Stable question ID: first 20 hex chars of SHA-256 over canonical
    question text + scope. Whitespace is collapsed so cosmetic edits do
    not mint a new question."""
    norm = " ".join(text.split())
    return content_hash({"text": norm, "scope": dict(scope)})[:QUESTION_ID_HEX_LEN]


def to_micro_score(score: float) -> int:
    """Quantize a raw ranking score to an integer micro-score.

    This happens BEFORE sorting or serialization; every downstream
    comparison uses the integer. Ties are broken by canonical public key
    at the sort site, never by the discarded float tail.
    """
    if score != score or score in (float("inf"), float("-inf")):
        raise CanonicalizationError(f"non-finite score: {score!r}")
    return round(score * MICRO)


def format_micro_score(micro: int) -> str:
    """Exact 6-decimal display string built from the integer (no float
    round-trip, so '0.1' can never render as '0.100000000000000005')."""
    sign = "-" if micro < 0 else ""
    whole, frac = divmod(abs(int(micro)), MICRO)
    return f"{sign}{whole}.{frac:06d}"


def deterministic_runtime(*, torch_setup: bool = False) -> dict[str, Any]:
    """Pin the compile-path runtime; return the settings dict for the manifest.

    ``PYTHONHASHSEED=0`` re-exec is delegated to the serving layer's
    existing discipline (``src.service.determinism``) so both paths share
    one mechanism. Torch pinning is opt-in because contracts/storage work
    must not pay a torch import.
    """
    from src.service.determinism import ensure_pythonhashseed

    ensure_pythonhashseed()
    settings: dict[str, Any] = {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        "torch_pinned": bool(torch_setup),
    }
    if torch_setup:
        import numpy as np
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        np.random.seed(0)
        settings.update({
            "device": "cpu",
            "torch_num_threads": 1,
            "torch_deterministic_algorithms": True,
            "torch_seed": 0,
            "numpy_seed": 0,
            "torch_version": torch.__version__,
        })
    return settings
