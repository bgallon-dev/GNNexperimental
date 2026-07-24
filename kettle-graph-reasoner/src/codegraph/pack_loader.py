r"""Python reader for the ``kgr_pack`` binary corpus format (v0.2).

mmaps the per-pack files and yields jsonl-equivalent dicts that drop
straight into the existing ``ingest.build_npz`` and
``cases.load_repo_cases`` consumers. No content transformation — only
format conversion.

Pack format reference: ``C:/Users/Benjamin/Desktop/Tutorstructure/kgr_pack/src/pack_schema.odin``.
Magic ``KGRPACK\0``, little-endian, fixed-width.

Why mmap: the production pack is ~1 GB and the strings table alone is
422 MB. Reading whole files into memory means 1 GB of Python bytes per
process; mmap lets the kernel page in only what we touch and shares
across processes if we ever go multi-worker.
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Enum tables — MUST match values + ordering in
# C:/Users/Benjamin/Desktop/Tutorstructure/kgr_pack/src/schema.odin
# (1-based for fields with UNKNOWN at 0; 0-based for task / family /
# label_set / split). Mismatches here are silent data corruption.
# ---------------------------------------------------------------------------

NODE_KIND_VALUES = (
    "Assignment", "CallSite", "Class", "ExternalPackage", "ExternalSymbol",
    "Function", "Import", "Method", "Module", "ParseError", "Repository", "Return",
)
NATIVE_RELATION_VALUES = (
    "ASSIGNS", "CALLS", "CONTAINS", "DEFINES", "DEPENDS_ON_PACKAGE",
    "IMPORTS", "IMPORTS_RAW", "INHERITS_FROM", "RESOLVES_TO", "RETURNS",
)
META_RELATION_VALUES = (
    "CONTAINS", "DEPENDS_ON", "DERIVES_FROM", "NONE", "PRODUCES", "TRANSFORMS",
)
PROVENANCE_MODE_VALUES = (
    "EXTRACTED_FROM_AST", "INFERRED_BY_RESOLVER",
)
RESOLUTION_STATUS_VALUES = (
    "AMBIGUOUS", "DYNAMIC_UNRESOLVED", "EXTERNAL", "PARTIALLY_RESOLVED",
    "RESOLVED", "UNRESOLVED",
)
TASK_TYPE_VALUES = (  # 0-based
    "CALL_TARGET_RANKING", "MISSING_CALL_EDGE", "FUNCTION_DEPENDENCY_RANKING",
    "IMPORT_DEPENDENCY_RANKING", "SAME_MODULE_HARD_NEGATIVE",
    "REVERSE_DEPENDENCY_RANKING", "MISSING_CALLSITE_BRIDGE",
    "PARENT_SCOPE_RANKING", "CHILD_SCOPE_RANKING", "PACKAGE_DEPENDENCY_RANKING",
    "CALL_PATH_RANKING", "ABSTAIN_TARGET_RANKING",
    "CALL_DIRECTION_CLASSIFICATION", "UNRESOLVED_CALL_CLASSIFICATION",
    "NATIVE_RELATION_CLASSIFICATION",
)
TASK_FAMILY_VALUES = ("ranking", "abstain_ranking", "classification")
LABEL_SET_NAMES = ("", "CALL_DIRECTION", "UNRESOLVED_CALL", "NATIVE_RELATION")
SPLIT_VALUES = ("Train", "Val", "Test")  # 1-based; 0 = UNKNOWN

# Sentinel constants from pack_schema.odin.
NO_QUERY_NODE2 = 0xFFFFFFFF
ABSTAIN_ROW_PACKED = 0xFFFFFFFE
NULL_STR_IDX = 0xFFFFFFFF
NO_LABEL = -1
ABSTAIN_SENTINEL = "__ABSTAIN__"

# Per-record sizes (matches schema.odin / pack_schema.odin constants).
PACK_HEADER_BYTES = 96
REPO_ROW_BYTES = 32
NODE_HOT_BYTES = 8
NODE_META_BYTES = 28
EDGE_HOT_BYTES = 20
EDGE_META_BYTES = 22
CASE_RECORD_BYTES = 48
STRINGS_IDX_BYTES = 12
CASEIDX_ELEM_BYTES = 8


def _enum_or_unknown(values: tuple[str, ...], idx: int) -> str:
    """1-based enum lookup with 0 reserved as UNKNOWN ('')."""
    if idx == 0:
        return ""
    j = idx - 1
    if 0 <= j < len(values):
        return values[j]
    return ""


def _enum_strict(values: tuple[str, ...], idx: int) -> str:
    if 0 <= idx < len(values):
        return values[idx]
    return ""


# ---------------------------------------------------------------------------
# PackContext: open + mmap + parse header + repos
# ---------------------------------------------------------------------------

@dataclass
class _RepoSummary:
    idx: int
    name: str
    repo_id: str
    n_nodes: int
    n_edges: int
    n_cases: int


class PackContext:
    """Open a kgr_pack v0.2 directory; mmap the on-disk arrays.

    ``ctx = PackContext(pack_dir)`` then use the iterator methods to
    pull jsonl-equivalent dicts per repo. ``ctx.close()`` releases the
    mmaps; ``with`` works too.
    """

    def __init__(self, pack_dir: Path | str) -> None:
        self.dir = Path(pack_dir)
        if not (self.dir / "header.bin").is_file():
            raise FileNotFoundError(f"missing header.bin in {self.dir}")
        self._mm: list[mmap.mmap] = []
        self._files: list = []

        # --- header ---
        hdr = (self.dir / "header.bin").read_bytes()
        if hdr[:8] != b"KGRPACK\x00":
            raise ValueError(f"bad magic in header.bin: {hdr[:8]!r}")
        ver_maj, ver_min = struct.unpack_from("<HH", hdr, 8)
        if (ver_maj, ver_min) != (0, 2):
            raise ValueError(
                f"unsupported pack version {ver_maj}.{ver_min} "
                f"(loader requires v0.2; rebuild the pack with the v0.2 packer)"
            )
        self.endian = hdr[12]
        self.id_width = hdr[13]
        self.flags, = struct.unpack_from("<H", hdr, 14)
        self.repo_count, = struct.unpack_from("<I", hdr, 16)
        (self.node_count, self.edge_count, self.case_count,
         self.string_count, self.case_list_elems) = struct.unpack_from(
            "<QQQQQ", hdr, 20
        )
        self.pack_mode, = struct.unpack_from("<I", hdr, 60)
        self.has_meta = bool(self.flags & 0x0002)
        self.has_strings = bool(self.flags & 0x0001)
        if not self.has_meta:
            raise ValueError(
                "pack is train-min (no nodes_meta / edges_meta / strings). "
                "The harness needs original node id strings to seed the "
                "identity vector deterministically — repack in train-debug "
                "mode (kgr-pack pack ... --mode train-debug) or use --mode "
                "both."
            )

        # --- mmap all required files ---
        self._mm_nodes_hot = self._open_mmap("nodes_hot.bin")
        self._mm_nodes_meta = self._open_mmap("nodes_meta.bin")
        self._mm_edges_hot = self._open_mmap("edges_hot.bin")
        self._mm_edges_meta = self._open_mmap("edges_meta.bin")
        self._mm_cases = self._open_mmap("cases.bin")
        self._mm_case_lists = self._open_mmap("case_lists.bin")
        self._mm_strings = self._open_mmap("strings.bin")
        self._mm_strings_idx = self._open_mmap("strings.idx")
        self._mm_repos = self._open_mmap("repos.bin")

        # --- repos.bin → list of RepoSummary ---
        self.repos: list[_RepoSummary] = []
        for i in range(self.repo_count):
            off = i * REPO_ROW_BYTES
            (id_off, name_off, _commit_off, _date_off,
             n_nodes, n_edges, n_cases, _) = struct.unpack_from(
                "<IIIIIIII", self._mm_repos, off
            )
            self.repos.append(_RepoSummary(
                idx=i,
                name=self._read_string(name_off),
                repo_id=self._read_string(id_off),
                n_nodes=int(n_nodes), n_edges=int(n_edges), n_cases=int(n_cases),
            ))

        # Sanity: total per-repo counts must match the header's globals.
        ts = sum(r.n_nodes for r in self.repos)
        if ts != self.node_count:
            raise ValueError(
                f"repos.bin sum {ts} != header.node_count {self.node_count}"
            )

        # Pre-compute per-repo node/edge/case start offsets (cumulative).
        # The packer writes per-repo in repo_idx order, so dense IDs are
        # ordered: repo 0's ids first, then repo 1, etc.
        self._node_start = [0]
        self._edge_start = [0]
        self._case_start = [0]
        for r in self.repos:
            self._node_start.append(self._node_start[-1] + r.n_nodes)
            self._edge_start.append(self._edge_start[-1] + r.n_edges)
            self._case_start.append(self._case_start[-1] + r.n_cases)

    # ----- internal helpers -----

    def _open_mmap(self, name: str) -> mmap.mmap:
        p = self.dir / name
        if not p.is_file():
            raise FileNotFoundError(f"missing {name} in pack dir {self.dir}")
        f = open(p, "rb")
        self._files.append(f)
        # Read-only mmap; size 0 lets the kernel pick the whole file.
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._mm.append(m)
        return m

    def _read_string(self, off: int) -> str:
        if off == NULL_STR_IDX:
            return ""
        rec_off = off * STRINGS_IDX_BYTES
        s_off, s_len = struct.unpack_from("<QI", self._mm_strings_idx, rec_off)
        return self._mm_strings[s_off:s_off + s_len].decode(
            "utf-8", errors="replace"
        )

    def _read_dense_node_id(self, dense_idx: int) -> str:
        """Return the original-id string for a global dense node index."""
        if dense_idx == ABSTAIN_ROW_PACKED:
            return ABSTAIN_SENTINEL
        meta_off = dense_idx * NODE_META_BYTES
        # nodes_meta layout: name_str, qname_str, file_str, original_id_str, ...
        oid_off, = struct.unpack_from("<I", self._mm_nodes_meta, meta_off + 12)
        return self._read_string(oid_off)

    def _read_dense_edge_id(self, dense_idx: int) -> str:
        # edges_meta layout: file_str, sl, sc, callsite, original_id_str, resmeth
        meta_off = dense_idx * EDGE_META_BYTES
        # file_str (u32) + sl (u32) + sc (u16) + callsite (u32) = 14; oid at +14
        oid_off, = struct.unpack_from("<I", self._mm_edges_meta, meta_off + 14)
        return self._read_string(oid_off)

    # ----- public API -----

    def close(self) -> None:
        for m in self._mm:
            try:
                m.close()
            except Exception:
                pass
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self._mm.clear()
        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def iter_repos(self) -> Iterator[_RepoSummary]:
        yield from self.repos

    def get_repo_by_name(self, name: str) -> _RepoSummary | None:
        for r in self.repos:
            if r.name == name:
                return r
        return None

    # ---- jsonl-equivalent per-repo iterators ----

    def iter_nodes(self, repo_idx: int) -> Iterator[dict]:
        """Yield jsonl-shaped node dicts for a repo. Field names match
        what ingest.py and cases.py read from nodes.jsonl."""
        start = self._node_start[repo_idx]
        end = self._node_start[repo_idx + 1]
        repo_id = self.repos[repo_idx].repo_id
        for dense in range(start, end):
            hot_off = dense * NODE_HOT_BYTES
            rp_idx, kind_id, flags = struct.unpack_from(
                "<IBB", self._mm_nodes_hot, hot_off
            )
            if rp_idx != repo_idx:
                # Should not happen if pack is consistent.
                continue
            meta_off = dense * NODE_META_BYTES
            (name_idx, qname_idx, fp_idx, oid_idx,
             start_line, end_line, start_col, end_col) = struct.unpack_from(
                "<IIIIIIHH", self._mm_nodes_meta, meta_off
            )
            file_path = self._read_string(fp_idx)
            yield {
                "id": self._read_string(oid_idx),
                "kind": _enum_or_unknown(NODE_KIND_VALUES, kind_id),
                "repo_id": repo_id,
                "name": self._read_string(name_idx),
                "qualified_name": self._read_string(qname_idx),
                # module_name is only read by task_v2_generator (which
                # runs on jsonl, not packs). Derive from file_path so
                # the field is correct if anyone inspects it.
                "module_name": _module_from_file_path(file_path),
                "file_path": file_path,
                "start_line": int(start_line),
                "end_line": int(end_line),
                "start_col": int(start_col),
                "end_col": int(end_col),
                # Fields ingest.py reads for routing logic:
                "is_method": bool(flags & 0x02),
                "is_test": bool(flags & 0x04),
                # The fields below are needed for case generation (e.g.,
                # callee_resolved_id for abstain task) but aren't stored in
                # the pack's nodes_meta — they live in edges (RESOLVES_TO).
                # The harness only reads what build_npz needs; cases come
                # from cases.bin, not regenerated, so the missing fields
                # don't matter for the load-not-regen path.
                "callee_resolved_id": "",
                "resolution_status": "",
            }

    def iter_edges(self, repo_idx: int) -> Iterator[dict]:
        """Yield jsonl-shaped edge dicts for a repo."""
        start = self._edge_start[repo_idx]
        end = self._edge_start[repo_idx + 1]
        repo_id = self.repos[repo_idx].repo_id
        for dense in range(start, end):
            hot_off = dense * EDGE_HOT_BYTES
            (src, tgt, native_id, meta_id, prov_id, res_id,
             rp_idx, conf_q, eflags) = struct.unpack_from(
                "<IIBBBBIHH", self._mm_edges_hot, hot_off
            )
            if rp_idx != repo_idx:
                continue
            meta_off = dense * EDGE_META_BYTES
            (fp_idx, sl, sc, callsite_idx, oid_idx,
             resmeth_idx) = struct.unpack_from(
                "<IIHIII", self._mm_edges_meta, meta_off
            )
            # Source/target are global dense indices; the harness's
            # build_npz expects original-id strings (which it then resolves
            # to its own per-repo row indices). Resolve dense → original.
            yield {
                "id": self._read_string(oid_idx),
                "source_id": self._read_dense_node_id(src),
                "target_id": self._read_dense_node_id(tgt),
                "native_relation": _enum_or_unknown(NATIVE_RELATION_VALUES, native_id),
                "meta_relation": _enum_or_unknown(META_RELATION_VALUES, meta_id),
                "repo_id": repo_id,
                "file_path": self._read_string(fp_idx),
                "start_line": int(sl),
                "start_col": int(sc),
                "provenance_mode": _enum_or_unknown(PROVENANCE_MODE_VALUES, prov_id),
                "resolution_status": _enum_or_unknown(RESOLUTION_STATUS_VALUES, res_id),
                "confidence": conf_q / 65535.0,
                "callsite_id": self._read_dense_node_id(callsite_idx)
                if callsite_idx != NULL_STR_IDX else "",
                "resolution_method": self._read_string(resmeth_idx),
                "valid": bool(eflags & 0x0001),
            }

    def iter_cases(self, repo_idx: int) -> Iterator[dict]:
        """Yield jsonl-shaped case dicts for a repo. Cases produced here
        match the shape of training_cases.jsonl + training_cases_v2.jsonl
        merged — v0.2 fields (task_family, label, label_set,
        query_node_ids_extra) are always populated."""
        start = self._case_start[repo_idx]
        end = self._case_start[repo_idx + 1]
        repo_id = self.repos[repo_idx].repo_id
        for dense in range(start, end):
            case_off = dense * CASE_RECORD_BYTES
            buf = self._mm_cases[case_off:case_off + CASE_RECORD_BYTES]
            task_id = buf[0]
            split_id = buf[1]
            family_id = buf[2]
            label_set_id = buf[3]
            label, = struct.unpack_from("<h", buf, 4)
            difficulty, = struct.unpack_from("<H", buf, 6)
            (rp_idx, query_node, query_node2, pos_off) = struct.unpack_from(
                "<IIII", buf, 8
            )
            pos_len, = struct.unpack_from("<H", buf, 24)
            neg_off, = struct.unpack_from("<I", buf, 26)
            neg_len, = struct.unpack_from("<H", buf, 30)
            req_off, = struct.unpack_from("<I", buf, 32)
            req_len, = struct.unpack_from("<H", buf, 36)
            gen_str_idx, = struct.unpack_from("<I", buf, 38)
            oid_str_idx, = struct.unpack_from("<I", buf, 42)
            flags, = struct.unpack_from("<H", buf, 46)
            if rp_idx != repo_idx:
                continue

            task_type = _enum_strict(TASK_TYPE_VALUES, task_id)
            split = _enum_or_unknown(SPLIT_VALUES, split_id)
            task_family = _enum_strict(TASK_FAMILY_VALUES, family_id)
            label_set = LABEL_SET_NAMES[label_set_id] if 0 <= label_set_id < len(LABEL_SET_NAMES) else ""

            # Read case_lists.bin slices for pos/neg/required.
            pos_ids = [
                self._read_dense_node_id(u32)
                for u32 in self._read_list(pos_off, pos_len)
            ]
            neg_ids = [
                self._read_dense_node_id(u32)
                for u32 in self._read_list(neg_off, neg_len)
            ]
            req_ids = [
                self._read_dense_edge_id(u32)
                for u32 in self._read_list(req_off, req_len)
            ]

            d = {
                "case_id": self._read_string(oid_str_idx),
                "task_type": task_type,
                "repo_id": repo_id,
                "query_node_id": self._read_dense_node_id(query_node),
                "positive_node_ids": pos_ids,
                "hard_negative_node_ids": neg_ids,
                "required_edge_ids": req_ids,
                "generation_rule": self._read_string(gen_str_idx),
                "split": split,
                "difficulty": int(difficulty),
                "source_pass": "V2_GENERATION" if (flags & 0x0001) else "TRAINING_GENERATION",
            }
            # v0.2 optional fields — always populated from the binary.
            if task_family:
                d["task_family"] = task_family
            if query_node2 != NO_QUERY_NODE2:
                d["query_node_ids_extra"] = [self._read_dense_node_id(query_node2)]
            if task_family == "classification" and label != NO_LABEL:
                d["label"] = int(label)
                d["label_set"] = label_set
            yield d

    def _read_list(self, off_in_elems: int, length: int) -> list[int]:
        if length == 0:
            return []
        byte_off = off_in_elems * 4
        raw = self._mm_case_lists[byte_off:byte_off + length * 4]
        return list(struct.unpack(f"<{length}I", raw))


def _module_from_file_path(file_path: str) -> str:
    """Derive ``module_name`` from a node's ``file_path``. The pack
    stores file_path verbatim in nodes_meta, so this is exact (unlike a
    qualified-name heuristic). ``app/__init__.py`` → ``app``;
    ``myapp/utils.py`` → ``myapp.utils``; non-Python files unchanged.

    Used only by ``task_v2_generator`` (which runs on jsonl corpora,
    not packs). For pack-mode training the field is populated for
    consistency but never read."""
    if not file_path:
        return ""
    p = file_path.replace("\\", "/")
    if p.endswith("/__init__.py"):
        p = p[: -len("/__init__.py")]
    elif p.endswith(".py"):
        p = p[:-3]
    return p.replace("/", ".")


# ---------------------------------------------------------------------------
# Drop-in helpers that mirror jsonl path
# ---------------------------------------------------------------------------

def discover_pack(pack_dir: Path | str) -> PackContext:
    """Open a pack and validate it's loadable. Returns the context;
    caller is responsible for ``close()`` (or use as ``with``)."""
    return PackContext(pack_dir)
