r"""Per-repo training-case loading + candidate pools (v0.2 schema).

Multi-repo aware: each repo is its own graph, so cases are resolved
against *that repo's* node ids and scored against *that repo's*
candidate pool. The Train/Val/Test split is NOT decided here — it is
assigned per fold by the harness (LORO-CV, fixed split, or file split).

v0.2 schema additions (all optional on disk — v0.1 cases parse via
``c.get(...)`` defaults):

* ``query_node_ids_extra`` — single-element list with the 2nd anchor
  for pair-anchored ranking tasks (MISSING_CALLSITE_BRIDGE,
  CALL_PATH_RANKING, CALL_DIRECTION_CLASSIFICATION,
  NATIVE_RELATION_CLASSIFICATION).
* ``task_family`` — ``"ranking"`` | ``"abstain_ranking"`` | ``"classification"``.
  Defaults to ``"ranking"`` for unannotated cases.
* ``label`` / ``label_set`` — classification only. Maps to a fixed
  label-vocabulary index (see ``LABEL_SETS``).
* ``positive_node_ids`` may contain the sentinel string ``"__ABSTAIN__"``
  for ABSTAIN_TARGET_RANKING; the loader maps it to row ``-1`` and the
  harness injects a learnable abstain embedding at scoring time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Stable task IDs. v0.1 keeps ids 0–4; v0.2 ranking 5–11; classification 12–14.
# Restore IMPORT_DEPENDENCY_RANKING (3) and SAME_MODULE_HARD_NEGATIVE (4)
# which were present in v0.1 jsonl but silently dropped before.
TASK_IDS = {
    # v0.1 — ranking
    "CALL_TARGET_RANKING": 0,
    "MISSING_CALL_EDGE": 1,
    "FUNCTION_DEPENDENCY_RANKING": 2,
    "IMPORT_DEPENDENCY_RANKING": 3,
    "SAME_MODULE_HARD_NEGATIVE": 4,
    # v0.2 — ranking
    "REVERSE_DEPENDENCY_RANKING": 5,
    "MISSING_CALLSITE_BRIDGE": 6,
    "PARENT_SCOPE_RANKING": 7,
    "CHILD_SCOPE_RANKING": 8,
    "PACKAGE_DEPENDENCY_RANKING": 9,
    "CALL_PATH_RANKING": 10,
    # v0.2 — abstain
    "ABSTAIN_TARGET_RANKING": 11,
    # v0.2 — classification
    "CALL_DIRECTION_CLASSIFICATION": 12,
    "UNRESOLVED_CALL_CLASSIFICATION": 13,
    "NATIVE_RELATION_CLASSIFICATION": 14,
}

# Inverse map for diagnostics.
TASK_NAMES = {v: k for k, v in TASK_IDS.items()}

# Default task -> family map (overridable per-case via the on-disk
# ``task_family`` field, useful if the generator ever wants to flag a
# task as a different family). The defaults match the canonical
# task→family assignment in the v0.2 plan.
DEFAULT_TASK_FAMILY = {
    # v0.1: all ranking
    "CALL_TARGET_RANKING": "ranking",
    "MISSING_CALL_EDGE": "ranking",
    "FUNCTION_DEPENDENCY_RANKING": "ranking",
    "IMPORT_DEPENDENCY_RANKING": "ranking",
    "SAME_MODULE_HARD_NEGATIVE": "ranking",
    # v0.2 ranking
    "REVERSE_DEPENDENCY_RANKING": "ranking",
    "MISSING_CALLSITE_BRIDGE": "ranking",
    "PARENT_SCOPE_RANKING": "ranking",
    "CHILD_SCOPE_RANKING": "ranking",
    "PACKAGE_DEPENDENCY_RANKING": "ranking",
    "CALL_PATH_RANKING": "ranking",
    # v0.2 abstain
    "ABSTAIN_TARGET_RANKING": "abstain_ranking",
    # v0.2 classification
    "CALL_DIRECTION_CLASSIFICATION": "classification",
    "UNRESOLVED_CALL_CLASSIFICATION": "classification",
    "NATIVE_RELATION_CLASSIFICATION": "classification",
}

# Classification label vocabularies (frozen, deterministic). Index 0 is
# the dominant class so the anchor "majority-class baseline" is
# well-defined and non-arbitrary.
LABEL_SETS: dict[str, tuple[str, ...]] = {
    "CALL_DIRECTION": ("A_TO_B", "B_TO_A", "BOTH", "NONE"),
    "UNRESOLVED_CALL": (
        "RESOLVED", "EXTERNAL", "PARTIALLY_RESOLVED",
        "AMBIGUOUS", "DYNAMIC_UNRESOLVED", "UNRESOLVED",
    ),
    "NATIVE_RELATION": (
        "CONTAINS", "DEFINES", "IMPORTS", "CALLS", "RESOLVES_TO",
        "ASSIGNS", "RETURNS", "INHERITS_FROM",
        "DEPENDS_ON_PACKAGE", "IMPORTS_RAW",
    ),
}
LABEL_SET_SIZES = {k: len(v) for k, v in LABEL_SETS.items()}

# Query vector layout (32-d). The shipped QueryToBall (query_dim=18) is
# never reloaded by harness.py — heads are always trained fresh — so
# expanding QUERY_DIM is free for our pipeline.
#
#   [0:3]    task-family one-hot (ranking, abstain_ranking, classification)
#   [3:18]   task ID one-hot (15 slots, one per TASK_IDS entry)
#   [18]     max_hops, normalized
#   [19]     pair_anchored flag (0/1)
#   [20:24]  reserved
#   [24:32]  anchor-1 identity vector (copy of x[query_row, 24:32])
#
# For pair-anchored tasks the harness concatenates the 2nd anchor's
# identity (8 dims of x[query_row2, 24:32]) onto query_vec at train +
# eval time, producing a 40-d effective input to the head — see
# harness._pair_anchor_concat. Heads instantiated with query_dim=32
# vs query_dim=40 are independent; mixing families per harness run is
# safe because the family is read off the Case.
QUERY_DIM = 32

ABSTAIN_SENTINEL = "__ABSTAIN__"
ABSTAIN_ROW = -1  # marker for abstain sentinel inside pos/hardneg lists

_FAMILY_BIT = {"ranking": 0, "abstain_ranking": 1, "classification": 2}


@dataclass
class Case:
    case_id: str
    repo: str
    task: str
    task_id: int
    task_family: str             # "ranking" | "abstain_ranking" | "classification"
    query_row: int
    query_row2: int              # -1 if no second anchor
    query_file: str
    pos_rows: list[int]          # may contain ABSTAIN_ROW (-1) for abstain task
    hardneg_rows: list[int]
    label: int                   # -1 unless task_family == "classification"
    label_set: str               # "" unless classification
    split: str
    query_vec: np.ndarray        # (32,) float32
    # 8-d second-anchor identity (x[query_row2, 24:32]) for pair-anchored
    # tasks; None otherwise. Harness concatenates query_vec || query_vec_extra
    # at use time, yielding a 40-d input for pair-anchored heads.
    query_vec_extra: np.ndarray | None = None


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_case_files(data_dir: Path):
    """Yield case dicts from training_cases.jsonl AND, if present, the
    v0.2 supplement training_cases_v2.jsonl. The order is v1 first then
    v2 (matters for deterministic case_id collisions — generator must
    not emit conflicting ids)."""
    p1 = Path(data_dir) / "training_cases.jsonl"
    p2 = Path(data_dir) / "training_cases_v2.jsonl"
    if p1.is_file():
        yield from _read_jsonl(p1)
    if p2.is_file():
        yield from _read_jsonl(p2)


def collect_required_edges(
    data_dir: Path, tasks: set[str], cases_iter=None,
) -> set[str]:
    """Answer edges to ablate from the encoder's graph for this repo.
    Reads both v0.1 and v0.2 case files, OR a pre-loaded iterator
    (e.g. ``pack_loader.PackContext.iter_cases``)."""
    out: set[str] = set()
    src = cases_iter if cases_iter is not None else _iter_case_files(Path(data_dir))
    for c in src:
        if c["task_type"] in tasks:
            out.update(c.get("required_edge_ids") or [])
    return out


def _encode_query(
    task_id: int,
    task_family: str,
    anchor_x: np.ndarray,
    anchor_x2: np.ndarray | None = None,
) -> np.ndarray:
    """Layout documented at the top of the module."""
    q = np.zeros(QUERY_DIM, np.float32)
    fam_bit = _FAMILY_BIT.get(task_family, 0)
    q[fam_bit] = 1.0
    if 0 <= task_id < 15:
        q[3 + task_id] = 1.0
    q[18] = 4.0 / 10.0                       # max_hops (tier1 convention)
    q[19] = 1.0 if anchor_x2 is not None else 0.0
    q[24:32] = anchor_x[24:32]
    # anchor2 identity is concatenated by the harness at use time, not
    # stored here, so the on-disk query_vec is always 32-d and the
    # ranking head's input dim is constant across non-pair-anchored
    # tasks. Pair-anchored tasks get a separate 40-d head.
    return q


def _resolve_row(graph, node_id) -> int | None:
    """Map a node id (string) or the abstain sentinel to a row index.
    Returns ``None`` for unknown ids (caller decides to skip)."""
    if node_id == ABSTAIN_SENTINEL:
        return ABSTAIN_ROW
    return graph.id_to_row.get(node_id)


def load_repo_cases(
    data_dir: Path,
    graph,
    x: np.ndarray,
    tasks: set[str],
    repo_name: str,
    cases_iter=None,
) -> tuple[list[Case], dict[str, np.ndarray], dict]:
    """Returns (cases, per-task corpus-wide pools, stats) for one repo.

    Reads both training_cases.jsonl (v0.1) and, when present,
    training_cases_v2.jsonl (v0.2). All v0.2 fields are optional in
    the on-disk dict; defaults preserve v0.1 behavior bit-for-bit.

    When ``cases_iter`` is provided (e.g. from
    ``pack_loader.PackContext.iter_cases``), the iterator is consumed
    in place of reading the jsonl files. The case dicts must have the
    same field set as the merged v0.1+v0.2 jsonl rows produce."""
    src = cases_iter if cases_iter is not None else _iter_case_files(Path(data_dir))
    raw = [c for c in src if c["task_type"] in tasks]
    cases: list[Case] = []
    skipped = 0
    for c in raw:
        task = c["task_type"]
        if task not in TASK_IDS:
            skipped += 1
            continue
        tid = TASK_IDS[task]
        family = c.get("task_family") or DEFAULT_TASK_FAMILY.get(task, "ranking")

        qid = c["query_node_id"]
        if qid not in graph.id_to_row:
            skipped += 1
            continue
        qrow = graph.id_to_row[qid]

        # Optional second anchor (pair-anchored ranking + classification).
        q2 = (c.get("query_node_ids_extra") or [None])[0]
        if q2 is not None:
            qrow2 = graph.id_to_row.get(q2)
            if qrow2 is None:
                skipped += 1
                continue
        else:
            qrow2 = -1

        if family == "classification":
            # No positive/hardneg lists for classification cases; the
            # ground-truth signal is the label.
            label_set = c.get("label_set", "")
            label = c.get("label")
            if label_set not in LABEL_SETS or not isinstance(label, int):
                skipped += 1
                continue
            if not 0 <= label < LABEL_SET_SIZES[label_set]:
                skipped += 1
                continue
            pos, neg = [], []
        else:
            pos_raw = c.get("positive_node_ids") or []
            neg_raw = c.get("hard_negative_node_ids") or []
            pos: list[int] = []
            for nid in pos_raw:
                r = _resolve_row(graph, nid)
                if r is not None:
                    pos.append(r)
            # Hardnegs go through _resolve_row too so the ABSTAIN sentinel
            # (used by commit-positive ABSTAIN_TARGET_RANKING cases as a
            # "don't pick abstain when a real target exists" distractor)
            # is preserved. Previously this used graph.id_to_row[nid]
            # directly, which silently dropped the sentinel and made the
            # head's abstain bias invisible at train time.
            neg: list[int] = []
            for nid in neg_raw:
                r = _resolve_row(graph, nid)
                if r is not None:
                    neg.append(r)
            if not pos:
                skipped += 1
                continue
            # Ranking tasks need hard-negs; abstain_ranking may have just
            # the abstain positive + non-abstain hardnegs (which IS hardneg-
            # like). For the abstain family we still require at least one
            # hardneg so the contrastive signal is non-degenerate.
            if not neg:
                skipped += 1
                continue
            label, label_set = -1, ""

        anchor_x = x[qrow]
        anchor_x2 = x[qrow2] if qrow2 != -1 else None
        qvec = _encode_query(tid, family, anchor_x, anchor_x2)
        qvec_extra = (
            anchor_x2[24:32].astype(np.float32).copy()
            if anchor_x2 is not None else None
        )

        cases.append(Case(
            case_id=c["case_id"],
            repo=repo_name,
            task=task,
            task_id=tid,
            task_family=family,
            query_row=qrow,
            query_row2=qrow2,
            query_file=graph.file_of.get(qid, ""),
            pos_rows=pos,
            hardneg_rows=neg,
            label=label,
            label_set=label_set,
            split="",
            query_vec=qvec,
            query_vec_extra=qvec_extra,
        ))

    # Pool computation: ranking + abstain only. Classification tasks have
    # no node-level pool. Excludes the abstain sentinel row from the pool
    # (the harness handles it via a separate learnable embedding).
    pool_kinds: dict[str, set[str]] = {t: set() for t in tasks}
    for c in cases:
        if c.task_family == "classification":
            continue
        for r in c.pos_rows + c.hardneg_rows:
            if r == ABSTAIN_ROW:
                continue
            pool_kinds[c.task].add(graph.kind_of[graph.node_ids[r]])
    pools: dict[str, np.ndarray] = {}
    for t, kinds in pool_kinds.items():
        rows = [
            graph.id_to_row[nid]
            for nid in graph.node_ids
            if graph.kind_of[nid] in kinds
        ]
        pools[t] = np.array(sorted(rows), dtype=np.int64)

    stats = {
        "n_cases": len(cases),
        "skipped": skipped,
        "task_counts": _task_counts(cases),
        "task_families": _family_counts(cases),
        "pool_sizes": {t: int(p.size) for t, p in pools.items()},
    }
    return cases, pools, stats


def assign_file_split(
    cases: list[Case],
    seed: int,
    fracs: tuple[float, float, float],
) -> None:
    """Assign train/val/test by source file (in place). Used both for
    the single-repo fallback (3-way) and to carve a held-out repo's
    cases into val/test (set fracs=(0.0, v, t))."""
    files = sorted({c.query_file for c in cases})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n = len(files)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    fsplit: dict[str, str] = {}
    for rank, fi in enumerate(perm):
        if rank < n_tr:
            fsplit[files[fi]] = "train"
        elif rank < n_tr + n_va:
            fsplit[files[fi]] = "val"
        else:
            fsplit[files[fi]] = "test"
    for c in cases:
        c.split = fsplit.get(c.query_file, "test")


def _task_counts(cases: list[Case]) -> dict:
    out: dict = {}
    for c in cases:
        out[c.task] = out.get(c.task, 0) + 1
    return out


def _family_counts(cases: list[Case]) -> dict:
    out: dict = {}
    for c in cases:
        out[c.task_family] = out.get(c.task_family, 0) + 1
    return out
