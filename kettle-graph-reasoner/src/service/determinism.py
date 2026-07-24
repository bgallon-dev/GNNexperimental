r"""Process determinism contract for the serving layer.

``scripts/neo4j_eval_export.py:_encode_graph`` derives the edge-type slot
order from ``sorted(set(et_names), key=-count)`` over rel-type STRINGS.
Python string hashing is randomized per process (PYTHONHASHSEED), so for
rel types with tied counts that set's iteration order -- and therefore the
baked edge-type one-hot slot assignment -- is process-dependent in the
reference itself. (Node-type ordering is already hash-stable: it is done
in integer cache-label-ID space; see ``tensor_contract``.)

To make the live serving path BIT-EXACT to the reference export pipeline
(verify.py P1) and reproducible run-to-run in production, the serving
process pins ``PYTHONHASHSEED=0`` -- the same determinism discipline the
project already applies elsewhere (e.g. ``torch.set_num_threads(1)`` in
the latency bench). This is a no-op once set; it re-execs once if not.
"""

from __future__ import annotations

import os
import subprocess
import sys

_SEED = "0"


def ensure_pythonhashseed() -> None:
    """Re-launch the current process once with ``PYTHONHASHSEED=0`` if it
    is not already pinned, then exit with the child's status. Safe to call
    at any entrypoint top; it is a no-op on the (re-launched) second entry.

    Uses ``subprocess`` rather than ``os.execv``: on Windows ``execv``
    re-joins argv into a command line and splits the (space-containing)
    interpreter/script path -- subprocess' list form quotes correctly on
    every platform.
    """
    if os.environ.get("PYTHONHASHSEED") == _SEED:
        return
    env = {**os.environ, "PYTHONHASHSEED": _SEED}
    raise SystemExit(
        subprocess.run([sys.executable, *sys.argv], env=env).returncode)


def hashseed_ok() -> bool:
    return os.environ.get("PYTHONHASHSEED") == _SEED


def child_env() -> dict:
    """Environment for spawned reference subprocesses (pins the same seed
    so a regenerated reference is deterministic and parity-comparable)."""
    return {**os.environ, "PYTHONHASHSEED": _SEED}
