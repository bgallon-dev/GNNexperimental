r"""v0.2 code-graph task generator.

Post-processor over a repo's ``nodes.jsonl`` + ``edges.jsonl`` (plus,
optionally, ``training_cases.jsonl`` for context) that emits the 10
v0.2 task types as supplementary cases written to
``training_cases_v2.jsonl`` alongside the v0.1 file.

The 10 tasks split into three families (matching ``cases.DEFAULT_TASK_FAMILY``):

* **ranking** (6): REVERSE_DEPENDENCY_RANKING, MISSING_CALLSITE_BRIDGE,
  PARENT_SCOPE_RANKING, CHILD_SCOPE_RANKING, PACKAGE_DEPENDENCY_RANKING,
  CALL_PATH_RANKING.
* **abstain_ranking** (1): ABSTAIN_TARGET_RANKING.
* **classification** (3): CALL_DIRECTION_CLASSIFICATION,
  UNRESOLVED_CALL_CLASSIFICATION, NATIVE_RELATION_CLASSIFICATION.

Deterministic in ``--seed``. Reuses ``ingest._read_jsonl`` for I/O.

CLI:

    python -m src.codegraph.task_v2_generator \
        --corpus-root ../corpus_validation \
        --extra-repo ../tutorstructure_patch \
        --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .cases import LABEL_SETS
from .ingest import _read_jsonl

ABSTAIN_SENTINEL = "__ABSTAIN__"


# ---------------------------------------------------------------------------
# Per-repo index (one-time build over nodes + edges)
# ---------------------------------------------------------------------------

@dataclass
class RepoIndex:
    repo_id: str
    nodes_by_id: dict[str, dict] = field(default_factory=dict)
    kind_buckets: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    edges: list[dict] = field(default_factory=list)
    # Adjacency by native_relation, both directions.
    out_by_rel: dict[str, dict[str, list[dict]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    in_by_rel: dict[str, dict[str, list[dict]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list)))


def _build_index(repo_dir: Path) -> RepoIndex:
    idx = RepoIndex(repo_id="")
    for n in _read_jsonl(repo_dir / "nodes.jsonl"):
        idx.nodes_by_id[n["id"]] = n
        idx.kind_buckets[n["kind"]].append(n["id"])
        if not idx.repo_id and n.get("repo_id"):
            idx.repo_id = n["repo_id"]
    for e in _read_jsonl(repo_dir / "edges.jsonl"):
        idx.edges.append(e)
        rel = e["native_relation"]
        idx.out_by_rel[rel][e["source_id"]].append(e)
        idx.in_by_rel[rel][e["target_id"]].append(e)
    return idx


# ---------------------------------------------------------------------------
# Case-id helper (stable across runs given (repo_id, task_type, key))
# ---------------------------------------------------------------------------

def _case_id(repo_id: str, task: str, key: str) -> str:
    h = hashlib.blake2b(
        f"{repo_id}|{task}|{key}".encode(), digest_size=12
    ).hexdigest()
    return f"case_{h}"


# ---------------------------------------------------------------------------
# Per-task generators
# ---------------------------------------------------------------------------

def _gen_reverse_dependency_ranking(
    idx: RepoIndex, rng: random.Random, n_hard_neg: int
) -> list[dict]:
    """For each Function/Method with ≥1 caller, emit a case where the
    positives are the calling Functions/Methods.

    Caller derivation: a CallSite ``cs`` has ``callee_resolved_id == f``
    (when resolved), and is contained by some enclosing scope. The
    enclosing Function/Method (via ``enclosing_id`` chain) is the
    caller. We use ``cs.enclosing_id`` directly (CallSites have a
    nearest enclosing scope) and walk up if it's not a Function/Method.
    """
    out: list[dict] = []
    fn_kinds = {"Function", "Method"}
    callsites_by_target: dict[str, list[str]] = defaultdict(list)
    for cs_id in idx.kind_buckets.get("CallSite", []):
        cs = idx.nodes_by_id[cs_id]
        tgt = cs.get("callee_resolved_id") or ""
        if not tgt or tgt not in idx.nodes_by_id:
            continue
        if idx.nodes_by_id[tgt]["kind"] not in fn_kinds:
            continue
        enc = cs.get("enclosing_id") or ""
        while enc and enc in idx.nodes_by_id and idx.nodes_by_id[enc]["kind"] not in fn_kinds:
            enc = idx.nodes_by_id[enc].get("enclosing_id") or ""
        if enc and enc in idx.nodes_by_id and idx.nodes_by_id[enc]["kind"] in fn_kinds:
            if enc != tgt:                   # don't count self-calls as callers
                callsites_by_target[tgt].append(enc)

    fn_ids_by_module: dict[str, list[str]] = defaultdict(list)
    for fid in idx.kind_buckets.get("Function", []) + idx.kind_buckets.get("Method", []):
        fn_ids_by_module[idx.nodes_by_id[fid].get("module_name", "")].append(fid)

    # Cap positives per case. Popular utility functions can have hundreds
    # of callers across a big repo (e.g. helpers in numpy/scipy); without
    # this cap, individual cases blow up per-case work in the harness.
    POS_CAP = 16
    for tgt, callers in callsites_by_target.items():
        callers = sorted(set(callers))
        if not callers:
            continue
        if len(callers) > POS_CAP:
            callers = sorted(rng.sample(callers, POS_CAP))
        mod = idx.nodes_by_id[tgt].get("module_name", "")
        same_mod = [f for f in fn_ids_by_module.get(mod, []) if f != tgt and f not in callers]
        if len(same_mod) < n_hard_neg:
            other = [f for f in fn_ids_by_module.values() for f in f
                     if isinstance(f, str) and f != tgt and f not in callers and f not in same_mod]
            same_mod = same_mod + other
        hard_negs = same_mod[:n_hard_neg] if len(same_mod) >= n_hard_neg else same_mod
        if not hard_negs:
            continue
        out.append({
            "case_id": _case_id(idx.repo_id, "REVERSE_DEPENDENCY_RANKING", tgt),
            "task_type": "REVERSE_DEPENDENCY_RANKING",
            "task_family": "ranking",
            "repo_id": idx.repo_id,
            "query_node_id": tgt,
            "positive_node_ids": callers,
            "hard_negative_node_ids": hard_negs,
            "required_edge_ids": [],
            "generation_rule": "reverse_callers_via_enclosing_scope",
            "split": "Train",
            "difficulty": 2,
            "source_pass": "V2_GENERATION",
        })
    return out


def _gen_missing_callsite_bridge(
    idx: RepoIndex, rng: random.Random, n_hard_neg: int
) -> list[dict]:
    """For each chain f_A -CONTAINS-> cs -RESOLVES_TO-> f_B emit a case
    where query=(f_A, f_B), positive=cs. Hard negatives = other
    CallSites resolving to a different target OR contained in a
    different scope but resolving to f_B."""
    fn_kinds = {"Function", "Method"}
    out: list[dict] = []
    cs_ids = idx.kind_buckets.get("CallSite", [])
    # Index all RESOLVED callsites by (enclosing-fn, resolved-target).
    chains: dict[tuple[str, str], list[str]] = defaultdict(list)
    cs_resolving_to: dict[str, list[str]] = defaultdict(list)
    cs_inside_fn: dict[str, list[str]] = defaultdict(list)
    for cs_id in cs_ids:
        cs = idx.nodes_by_id[cs_id]
        tgt = cs.get("callee_resolved_id") or ""
        if not tgt or tgt not in idx.nodes_by_id:
            continue
        if idx.nodes_by_id[tgt]["kind"] not in fn_kinds:
            continue
        enc = cs.get("enclosing_id") or ""
        while enc and enc in idx.nodes_by_id and idx.nodes_by_id[enc]["kind"] not in fn_kinds:
            enc = idx.nodes_by_id[enc].get("enclosing_id") or ""
        if not (enc and enc in idx.nodes_by_id and idx.nodes_by_id[enc]["kind"] in fn_kinds):
            continue
        chains[(enc, tgt)].append(cs_id)
        cs_resolving_to[tgt].append(cs_id)
        cs_inside_fn[enc].append(cs_id)

    for (f_a, f_b), bridge_css in chains.items():
        # Pick the first bridge as canonical positive (deterministic).
        positive = sorted(bridge_css)[0]
        # Hard-neg pool: other CSs inside f_A not resolving to f_B + other
        # CSs resolving to f_B not inside f_A.
        cand_inside = [c for c in cs_inside_fn[f_a] if c not in bridge_css]
        cand_resolving = [c for c in cs_resolving_to[f_b] if c not in bridge_css]
        rng.shuffle(cand_inside)
        rng.shuffle(cand_resolving)
        hard_negs: list[str] = []
        while (cand_inside or cand_resolving) and len(hard_negs) < n_hard_neg:
            if cand_inside:
                hard_negs.append(cand_inside.pop())
            if len(hard_negs) >= n_hard_neg:
                break
            if cand_resolving:
                hard_negs.append(cand_resolving.pop())
        if not hard_negs:
            continue
        out.append({
            "case_id": _case_id(idx.repo_id, "MISSING_CALLSITE_BRIDGE",
                                f"{f_a}|{f_b}"),
            "task_type": "MISSING_CALLSITE_BRIDGE",
            "task_family": "ranking",
            "repo_id": idx.repo_id,
            "query_node_id": f_a,
            "query_node_ids_extra": [f_b],
            "positive_node_ids": [positive],
            "hard_negative_node_ids": hard_negs,
            "required_edge_ids": [],
            "generation_rule": "hide_callsite_bridge",
            "split": "Train",
            "difficulty": 3,
            "source_pass": "V2_GENERATION",
        })
    return out


def _gen_parent_scope_ranking(
    idx: RepoIndex, rng: random.Random, n_hard_neg: int
) -> list[dict]:
    """For each node with a parent via CONTAINS or DEFINES, positive =
    parent node. Hard negatives = same-module sibling scopes."""
    out: list[dict] = []
    parent_of: dict[str, str] = {}
    for rel in ("CONTAINS", "DEFINES"):
        for src, edges in idx.out_by_rel.get(rel, {}).items():
            for e in edges:
                tgt = e["target_id"]
                if tgt in idx.nodes_by_id and tgt not in parent_of:
                    parent_of[tgt] = src
    scope_kinds = {"Repository", "Module", "Class", "Function", "Method"}
    scopes_by_module: dict[str, list[str]] = defaultdict(list)
    for nid, n in idx.nodes_by_id.items():
        if n["kind"] in scope_kinds:
            scopes_by_module[n.get("module_name", "")].append(nid)

    for node_id, parent in parent_of.items():
        mod = idx.nodes_by_id[node_id].get("module_name", "")
        siblings = [s for s in scopes_by_module.get(mod, []) if s != parent]
        if not siblings:
            continue
        rng.shuffle(siblings)
        hard_negs = siblings[:n_hard_neg]
        out.append({
            "case_id": _case_id(idx.repo_id, "PARENT_SCOPE_RANKING", node_id),
            "task_type": "PARENT_SCOPE_RANKING",
            "task_family": "ranking",
            "repo_id": idx.repo_id,
            "query_node_id": node_id,
            "positive_node_ids": [parent],
            "hard_negative_node_ids": hard_negs,
            "required_edge_ids": [],
            "generation_rule": "parent_via_contains_or_defines",
            "split": "Train",
            "difficulty": 1,
            "source_pass": "V2_GENERATION",
        })
    return out


def _gen_child_scope_ranking(
    idx: RepoIndex, rng: random.Random, n_hard_neg: int
) -> list[dict]:
    """For each scope, positives = its CONTAINS/DEFINES children. Hard
    negatives = children of sibling scopes within the same module."""
    out: list[dict] = []
    children_of: dict[str, list[str]] = defaultdict(list)
    for rel in ("CONTAINS", "DEFINES"):
        for src, edges in idx.out_by_rel.get(rel, {}).items():
            for e in edges:
                children_of[src].append(e["target_id"])

    children_by_module: dict[str, list[str]] = defaultdict(list)
    for sid, kids in children_of.items():
        mod = idx.nodes_by_id[sid].get("module_name", "")
        children_by_module[mod].extend(kids)

    # Cap positives per case. Module-scope queries on big repos (django,
    # pandas, scipy) have thousands of CONTAINS+DEFINES descendants when
    # the propagation reaches callsites/assignments/returns — uncapped,
    # individual cases produced unbounded per-case work in the harness
    # (~hours on Colab, cf. plans/tutorstructure-has-code-arranged-
    # sleepy-puppy.md). 16 is enough for nDCG@10 to remain meaningful.
    POS_CAP = 16
    for scope_id, kids in children_of.items():
        kids_set = set(kids)
        mod = idx.nodes_by_id[scope_id].get("module_name", "")
        candidates = [c for c in children_by_module.get(mod, [])
                      if c not in kids_set and c in idx.nodes_by_id]
        if not candidates:
            continue
        rng.shuffle(candidates)
        hard_negs = candidates[:n_hard_neg]
        positives = sorted(set(kids))
        if len(positives) > POS_CAP:
            positives = sorted(rng.sample(positives, POS_CAP))
        out.append({
            "case_id": _case_id(idx.repo_id, "CHILD_SCOPE_RANKING", scope_id),
            "task_type": "CHILD_SCOPE_RANKING",
            "task_family": "ranking",
            "repo_id": idx.repo_id,
            "query_node_id": scope_id,
            "positive_node_ids": positives,
            "hard_negative_node_ids": hard_negs,
            "required_edge_ids": [],
            "generation_rule": "children_via_contains_or_defines",
            "split": "Train",
            "difficulty": 1,
            "source_pass": "V2_GENERATION",
        })
    return out


def _gen_package_dependency_ranking(
    idx: RepoIndex, rng: random.Random, n_hard_neg: int
) -> list[dict]:
    """For each Module, positives = ExternalPackage neighbors via
    DEPENDS_ON_PACKAGE. Hard negatives = ExternalPackages other modules
    import but this one doesn't."""
    out: list[dict] = []
    pkgs = idx.kind_buckets.get("ExternalPackage", [])
    if not pkgs:
        return out
    for mod_id in idx.kind_buckets.get("Module", []):
        deps = [e["target_id"] for e in idx.out_by_rel.get("DEPENDS_ON_PACKAGE", {}).get(mod_id, [])
                if e["target_id"] in idx.nodes_by_id]
        if not deps:
            continue
        dep_set = set(deps)
        cand = [p for p in pkgs if p not in dep_set]
        if not cand:
            continue
        rng.shuffle(cand)
        hard_negs = cand[:n_hard_neg]
        out.append({
            "case_id": _case_id(idx.repo_id, "PACKAGE_DEPENDENCY_RANKING", mod_id),
            "task_type": "PACKAGE_DEPENDENCY_RANKING",
            "task_family": "ranking",
            "repo_id": idx.repo_id,
            "query_node_id": mod_id,
            "positive_node_ids": sorted(set(deps)),
            "hard_negative_node_ids": hard_negs,
            "required_edge_ids": [],
            "generation_rule": "depends_on_package_neighbors",
            "split": "Train",
            "difficulty": 1,
            "source_pass": "V2_GENERATION",
        })
    return out


def _gen_call_path_ranking(
    idx: RepoIndex, rng: random.Random, n_hard_neg: int, cap: int
) -> list[dict]:
    """For 2-hop function call paths src→f_mid→tgt, query=(src,tgt),
    positive=f_mid (the path is identified by its bridge). Hard negs =
    other functions reachable from src in 1 hop that don't reach tgt."""
    fn_kinds = {"Function", "Method"}
    # Build per-function direct-callee map by walking CALLS edges out
    # of each function's CallSites.
    callees_of: dict[str, set[str]] = defaultdict(set)
    for cs_id in idx.kind_buckets.get("CallSite", []):
        cs = idx.nodes_by_id[cs_id]
        tgt = cs.get("callee_resolved_id") or ""
        if not tgt or tgt not in idx.nodes_by_id:
            continue
        if idx.nodes_by_id[tgt]["kind"] not in fn_kinds:
            continue
        enc = cs.get("enclosing_id") or ""
        while enc and enc in idx.nodes_by_id and idx.nodes_by_id[enc]["kind"] not in fn_kinds:
            enc = idx.nodes_by_id[enc].get("enclosing_id") or ""
        if enc and enc in idx.nodes_by_id and idx.nodes_by_id[enc]["kind"] in fn_kinds:
            callees_of[enc].add(tgt)

    out: list[dict] = []
    for src, mids in callees_of.items():
        for mid in sorted(mids):
            mid_callees = callees_of.get(mid, set())
            for tgt in sorted(mid_callees):
                if tgt == src or tgt == mid:
                    continue
                if tgt in callees_of[src]:
                    continue  # direct edge — not interesting for path ranking
                # Hard negs: other immediate callees of src that don't
                # reach tgt in 1 hop.
                hn_pool = [
                    m for m in mids
                    if m != mid and tgt not in callees_of.get(m, set())
                ]
                if not hn_pool:
                    continue
                rng.shuffle(hn_pool)
                hard_negs = hn_pool[:n_hard_neg]
                out.append({
                    "case_id": _case_id(idx.repo_id, "CALL_PATH_RANKING",
                                        f"{src}|{tgt}|{mid}"),
                    "task_type": "CALL_PATH_RANKING",
                    "task_family": "ranking",
                    "repo_id": idx.repo_id,
                    "query_node_id": src,
                    "query_node_ids_extra": [tgt],
                    "positive_node_ids": [mid],
                    "hard_negative_node_ids": hard_negs,
                    "required_edge_ids": [],
                    "generation_rule": "two_hop_call_path_bridge",
                    "split": "Train",
                    "difficulty": 3,
                    "source_pass": "V2_GENERATION",
                })
                if len(out) >= cap:
                    return out
    return out


def _gen_abstain_target_ranking(
    idx: RepoIndex, rng: random.Random, n_hard_neg: int
) -> list[dict]:
    """Balanced abstain task. Two case families:

    * **Abstain-positive** — for unresolved / external / dynamic /
      partial CallSites, positive=``__ABSTAIN__``, hardnegs=lexical
      lures (same-module Functions whose name shares a token with
      ``callee_raw``). Teaches the head: when the encoder can't find a
      real target, choose abstain.
    * **Commit-positive** — for RESOLVED CallSites, positive=the actual
      ``callee_resolved_id`` and ``__ABSTAIN__`` is injected as a hard
      negative. Teaches the head: when a real target exists, DO NOT
      abstain.

    Without the second family the head trivially learns "always abstain"
    (observed: `loss=0.0000 rank_acc=1.000` after 1 epoch on the
    Colab smoke). The balance — sample at most as many RESOLVED cases
    as abstain cases so neither class dominates — keeps both signals
    in scope for the contrastive head."""
    out: list[dict] = []
    abstain_states = {"UNRESOLVED", "EXTERNAL", "DYNAMIC_UNRESOLVED",
                      "PARTIALLY_RESOLVED"}
    fn_kinds = {"Function", "Method"}
    fns_by_module: dict[str, list[str]] = defaultdict(list)
    for nid in idx.kind_buckets.get("Function", []) + idx.kind_buckets.get("Method", []):
        fns_by_module[idx.nodes_by_id[nid].get("module_name", "")].append(nid)

    def _lures_for(cs: dict) -> list[str]:
        callee_raw = (cs.get("callee_raw") or cs.get("name") or "").lower()
        if not callee_raw:
            return []
        tokens = {t for t in callee_raw.replace(".", "_").split("_") if len(t) >= 3}
        mod = cs.get("module_name", "")
        fns = list(fns_by_module.get(mod, []))
        rng.shuffle(fns)
        if not tokens:
            return fns[:n_hard_neg]
        lures: list[str] = []
        for fn_id in fns:
            name = (idx.nodes_by_id[fn_id].get("name", "") or "").lower()
            name_tokens = set(name.replace(".", "_").split("_"))
            if tokens & name_tokens:
                lures.append(fn_id)
                if len(lures) >= n_hard_neg:
                    break
        return lures if lures else fns[:n_hard_neg]

    # Collect abstain-positive and commit-positive case lists separately,
    # then balance.
    abstain_cases: list[dict] = []
    commit_cases: list[dict] = []
    for cs_id in idx.kind_buckets.get("CallSite", []):
        cs = idx.nodes_by_id[cs_id]
        status = cs.get("resolution_status", "")
        if status in abstain_states:
            lures = _lures_for(cs)
            if not lures:
                continue
            abstain_cases.append({
                "case_id": _case_id(idx.repo_id, "ABSTAIN_TARGET_RANKING",
                                    cs_id),
                "task_type": "ABSTAIN_TARGET_RANKING",
                "task_family": "abstain_ranking",
                "repo_id": idx.repo_id,
                "query_node_id": cs_id,
                "positive_node_ids": [ABSTAIN_SENTINEL],
                "hard_negative_node_ids": lures,
                "required_edge_ids": [],
                "generation_rule": "abstain_on_unresolved_callsite",
                "split": "Train",
                "difficulty": 3,
                "source_pass": "V2_GENERATION",
            })
        elif status == "RESOLVED":
            tgt = cs.get("callee_resolved_id") or ""
            if not tgt or tgt not in idx.nodes_by_id:
                continue
            if idx.nodes_by_id[tgt]["kind"] not in fn_kinds:
                continue
            lures = _lures_for(cs)
            # Drop the actual target from lures if it landed there;
            # ABSTAIN is the canonical first hard-neg so the head can
            # always learn "this case has a real target, not abstain."
            lures = [l for l in lures if l != tgt][:max(n_hard_neg - 1, 0)]
            hard_negs = [ABSTAIN_SENTINEL] + lures
            commit_cases.append({
                "case_id": _case_id(idx.repo_id, "ABSTAIN_TARGET_RANKING",
                                    cs_id),
                "task_type": "ABSTAIN_TARGET_RANKING",
                "task_family": "abstain_ranking",
                "repo_id": idx.repo_id,
                "query_node_id": cs_id,
                "positive_node_ids": [tgt],
                "hard_negative_node_ids": hard_negs,
                "required_edge_ids": [],
                "generation_rule": "commit_on_resolved_callsite",
                "split": "Train",
                "difficulty": 3,
                "source_pass": "V2_GENERATION",
            })

    # Balance: cap the larger class to match the smaller one. Both classes
    # are kept fully sampled if one dominates by a large factor — we want
    # the head to see both regimes regardless of corpus skew.
    rng.shuffle(abstain_cases)
    rng.shuffle(commit_cases)
    if len(abstain_cases) > 2 * len(commit_cases) and commit_cases:
        abstain_cases = abstain_cases[: 2 * len(commit_cases)]
    elif len(commit_cases) > 2 * len(abstain_cases) and abstain_cases:
        commit_cases = commit_cases[: 2 * len(abstain_cases)]
    out.extend(abstain_cases)
    out.extend(commit_cases)
    return out


def _gen_call_direction_classification(
    idx: RepoIndex, rng: random.Random, max_pairs: int
) -> list[dict]:
    """For (FuncA, FuncB) pairs from the call graph, classify direction:
    A→B / B→A / BOTH / NONE. Stratify so 'NONE' isn't oversampled."""
    fn_kinds = {"Function", "Method"}
    callees_of: dict[str, set[str]] = defaultdict(set)
    for cs_id in idx.kind_buckets.get("CallSite", []):
        cs = idx.nodes_by_id[cs_id]
        tgt = cs.get("callee_resolved_id") or ""
        if not tgt or tgt not in idx.nodes_by_id:
            continue
        if idx.nodes_by_id[tgt]["kind"] not in fn_kinds:
            continue
        enc = cs.get("enclosing_id") or ""
        while enc and enc in idx.nodes_by_id and idx.nodes_by_id[enc]["kind"] not in fn_kinds:
            enc = idx.nodes_by_id[enc].get("enclosing_id") or ""
        if enc and enc in idx.nodes_by_id and idx.nodes_by_id[enc]["kind"] in fn_kinds:
            callees_of[enc].add(tgt)

    all_fns = sorted(set(idx.kind_buckets.get("Function", [])) |
                     set(idx.kind_buckets.get("Method", [])))
    if len(all_fns) < 2:
        return []

    # Build pair → label.
    labels: list[tuple[str, str, str]] = []  # (a, b, label_name)
    seen = set()
    for a, bs in callees_of.items():
        for b in bs:
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            a_calls_b = b in callees_of.get(a, set())
            b_calls_a = a in callees_of.get(b, set())
            if a_calls_b and b_calls_a:
                lbl = "BOTH"
            elif a_calls_b:
                lbl = "A_TO_B"
            else:
                lbl = "B_TO_A"
            labels.append((key[0] == a and a or b,
                           key[0] == a and b or a, lbl))

    # NONE: sample same-module function pairs that don't call each other.
    fns_by_module: dict[str, list[str]] = defaultdict(list)
    for nid in all_fns:
        fns_by_module[idx.nodes_by_id[nid].get("module_name", "")].append(nid)
    none_target = max(1, len(labels) // 3)
    none_added = 0
    none_pairs: list[tuple[str, str, str]] = []
    modules = list(fns_by_module.values())
    rng.shuffle(modules)
    for fns in modules:
        if none_added >= none_target:
            break
        rng.shuffle(fns)
        for i in range(len(fns)):
            for j in range(i + 1, len(fns)):
                a, b = fns[i], fns[j]
                if b in callees_of.get(a, set()) or a in callees_of.get(b, set()):
                    continue
                none_pairs.append((a, b, "NONE"))
                none_added += 1
                if none_added >= none_target:
                    break
            if none_added >= none_target:
                break
    labels.extend(none_pairs)
    rng.shuffle(labels)
    if max_pairs:
        labels = labels[:max_pairs]

    out = []
    label_idx = {n: i for i, n in enumerate(LABEL_SETS["CALL_DIRECTION"])}
    for a, b, name in labels:
        out.append({
            "case_id": _case_id(idx.repo_id, "CALL_DIRECTION_CLASSIFICATION", f"{a}|{b}"),
            "task_type": "CALL_DIRECTION_CLASSIFICATION",
            "task_family": "classification",
            "repo_id": idx.repo_id,
            "query_node_id": a,
            "query_node_ids_extra": [b],
            "positive_node_ids": [],
            "hard_negative_node_ids": [],
            "required_edge_ids": [],
            "label": label_idx[name],
            "label_set": "CALL_DIRECTION",
            "generation_rule": "directionality_from_callsites",
            "split": "Train",
            "difficulty": 2,
            "source_pass": "V2_GENERATION",
        })
    return out


def _gen_unresolved_call_classification(
    idx: RepoIndex, rng: random.Random, max_per_class: int
) -> list[dict]:
    """Classify each CallSite's ``resolution_status``. Stratify so the
    dominant class isn't oversampled."""
    label_idx = {n: i for i, n in enumerate(LABEL_SETS["UNRESOLVED_CALL"])}
    by_class: dict[str, list[str]] = defaultdict(list)
    for cs_id in idx.kind_buckets.get("CallSite", []):
        status = idx.nodes_by_id[cs_id].get("resolution_status", "")
        if status in label_idx:
            by_class[status].append(cs_id)
        elif status == "AMBIGUOUS" or status == "PARTIALLY_RESOLVED":
            by_class[status].append(cs_id)
    for k in list(by_class):
        rng.shuffle(by_class[k])
        if max_per_class > 0:
            by_class[k] = by_class[k][:max_per_class]
    out: list[dict] = []
    for status, ids in by_class.items():
        if status not in label_idx:
            continue
        for cs_id in ids:
            out.append({
                "case_id": _case_id(idx.repo_id, "UNRESOLVED_CALL_CLASSIFICATION", cs_id),
                "task_type": "UNRESOLVED_CALL_CLASSIFICATION",
                "task_family": "classification",
                "repo_id": idx.repo_id,
                "query_node_id": cs_id,
                "positive_node_ids": [],
                "hard_negative_node_ids": [],
                "required_edge_ids": [],
                "label": label_idx[status],
                "label_set": "UNRESOLVED_CALL",
                "generation_rule": "resolution_status_label",
                "split": "Train",
                "difficulty": 2,
                "source_pass": "V2_GENERATION",
            })
    return out


def _gen_native_relation_classification(
    idx: RepoIndex, rng: random.Random, max_per_class: int
) -> list[dict]:
    """Classify each edge's ``native_relation``. Stratify so CONTAINS
    (the dominant class) doesn't drown the others."""
    label_idx = {n: i for i, n in enumerate(LABEL_SETS["NATIVE_RELATION"])}
    by_class: dict[str, list[dict]] = defaultdict(list)
    for e in idx.edges:
        rel = e.get("native_relation", "")
        if rel in label_idx:
            by_class[rel].append(e)
    for k in list(by_class):
        rng.shuffle(by_class[k])
        if max_per_class > 0:
            by_class[k] = by_class[k][:max_per_class]
    out: list[dict] = []
    for rel, edges in by_class.items():
        for e in edges:
            src, tgt = e["source_id"], e["target_id"]
            if src not in idx.nodes_by_id or tgt not in idx.nodes_by_id:
                continue
            out.append({
                "case_id": _case_id(idx.repo_id, "NATIVE_RELATION_CLASSIFICATION", e["id"]),
                "task_type": "NATIVE_RELATION_CLASSIFICATION",
                "task_family": "classification",
                "repo_id": idx.repo_id,
                "query_node_id": src,
                "query_node_ids_extra": [tgt],
                "positive_node_ids": [],
                "hard_negative_node_ids": [],
                "required_edge_ids": [],
                "label": label_idx[rel],
                "label_set": "NATIVE_RELATION",
                "generation_rule": "edge_native_relation_label",
                "split": "Train",
                "difficulty": 1,
                "source_pass": "V2_GENERATION",
            })
    return out


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

ALL_V2_TASKS = (
    "REVERSE_DEPENDENCY_RANKING",
    "MISSING_CALLSITE_BRIDGE",
    "PARENT_SCOPE_RANKING",
    "CHILD_SCOPE_RANKING",
    "PACKAGE_DEPENDENCY_RANKING",
    "CALL_PATH_RANKING",
    "ABSTAIN_TARGET_RANKING",
    "CALL_DIRECTION_CLASSIFICATION",
    "UNRESOLVED_CALL_CLASSIFICATION",
    "NATIVE_RELATION_CLASSIFICATION",
)


def generate_v2_for_repo(
    repo_dir: Path,
    *,
    seed: int = 0,
    n_hard_neg: int = 5,
    max_per_task: int = 0,
    tasks: set[str] | None = None,
) -> list[dict]:
    """Generate all v0.2 cases for a single repo. Deterministic given
    ``seed``. ``max_per_task=0`` means no cap (subject to per-generator
    intrinsic caps for stratified tasks)."""
    idx = _build_index(repo_dir)
    rng = random.Random(seed)
    requested = set(tasks) if tasks else set(ALL_V2_TASKS)
    out: list[dict] = []

    def _add(name: str, cases: list[dict]):
        if max_per_task > 0 and len(cases) > max_per_task:
            cases = cases[:max_per_task]
        out.extend(cases)

    if "REVERSE_DEPENDENCY_RANKING" in requested:
        _add("REVERSE_DEPENDENCY_RANKING",
             _gen_reverse_dependency_ranking(idx, rng, n_hard_neg))
    if "MISSING_CALLSITE_BRIDGE" in requested:
        _add("MISSING_CALLSITE_BRIDGE",
             _gen_missing_callsite_bridge(idx, rng, n_hard_neg))
    if "PARENT_SCOPE_RANKING" in requested:
        _add("PARENT_SCOPE_RANKING",
             _gen_parent_scope_ranking(idx, rng, n_hard_neg))
    if "CHILD_SCOPE_RANKING" in requested:
        _add("CHILD_SCOPE_RANKING",
             _gen_child_scope_ranking(idx, rng, n_hard_neg))
    if "PACKAGE_DEPENDENCY_RANKING" in requested:
        _add("PACKAGE_DEPENDENCY_RANKING",
             _gen_package_dependency_ranking(idx, rng, n_hard_neg))
    if "CALL_PATH_RANKING" in requested:
        path_cap = max_per_task if max_per_task > 0 else 2000
        _add("CALL_PATH_RANKING",
             _gen_call_path_ranking(idx, rng, n_hard_neg, path_cap))
    if "ABSTAIN_TARGET_RANKING" in requested:
        _add("ABSTAIN_TARGET_RANKING",
             _gen_abstain_target_ranking(idx, rng, n_hard_neg))
    if "CALL_DIRECTION_CLASSIFICATION" in requested:
        cdc_cap = max_per_task if max_per_task > 0 else 2000
        _add("CALL_DIRECTION_CLASSIFICATION",
             _gen_call_direction_classification(idx, rng, cdc_cap))
    if "UNRESOLVED_CALL_CLASSIFICATION" in requested:
        per_cls = max_per_task if max_per_task > 0 else 1000
        _add("UNRESOLVED_CALL_CLASSIFICATION",
             _gen_unresolved_call_classification(idx, rng, per_cls))
    if "NATIVE_RELATION_CLASSIFICATION" in requested:
        per_cls = max_per_task if max_per_task > 0 else 500
        _add("NATIVE_RELATION_CLASSIFICATION",
             _gen_native_relation_classification(idx, rng, per_cls))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_repos(corpus_root: Path, extra: list[str]) -> list[Path]:
    out: list[Path] = []
    if corpus_root.is_dir():
        out += sorted(p for p in corpus_root.iterdir()
                      if (p / "nodes.jsonl").is_file())
    for e in extra:
        p = Path(e)
        if (p / "nodes.jsonl").is_file():
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--extra-repo", action="append", default=[])
    ap.add_argument("--out-name", default="training_cases_v2.jsonl",
                    help="written next to nodes.jsonl in each repo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-hard-neg", type=int, default=5)
    ap.add_argument("--max-per-task", type=int, default=0,
                    help="0 = no cap (subject to per-generator intrinsic caps)")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="restrict to listed v0.2 task names; default=all")
    args = ap.parse_args()

    repos = _discover_repos(Path(args.corpus_root), args.extra_repo)
    if not repos:
        raise SystemExit(
            f"no repos with nodes.jsonl under {args.corpus_root} "
            f"or --extra-repo {args.extra_repo}"
        )
    requested = set(args.tasks) if args.tasks else None
    total = 0
    for repo in repos:
        cases = generate_v2_for_repo(
            repo, seed=args.seed, n_hard_neg=args.n_hard_neg,
            max_per_task=args.max_per_task, tasks=requested,
        )
        out_path = repo / args.out_name
        with open(out_path, "w", encoding="utf-8") as fh:
            for c in cases:
                fh.write(json.dumps(c) + "\n")
        by_task: dict[str, int] = {}
        for c in cases:
            by_task[c["task_type"]] = by_task.get(c["task_type"], 0) + 1
        total += len(cases)
        print(f"  {repo.name}: {len(cases)} cases -> {out_path.name}  "
              f"{by_task}")
    print(f"\nwrote {total} v0.2 cases across {len(repos)} repos")


if __name__ == "__main__":
    main()
