r"""Config-driven Neo4j label/rel-type -> KGR contract mapper.

This module is the *de-hardcoding* of the schema mapping that was inlined
in ``scripts/neo4j_eval_export.py`` (``_AUX_LABELS`` :91, ``_node_layer``
:125, ``_edge_category`` :129). The mapping now lives in
``schema_map.yaml`` (a descriptor, not code), satisfying CLAUDE.md
non-negotiable #3 (schema-portable: no hardcoded domain type embeddings --
to retarget a domain you edit the YAML, never the encoder).

The shipped ``schema_map.yaml`` defaults are byte-identical in behaviour
to the hardcoded reference, so the refactor is a provable no-op. Gate P0
(``verify.py`` / ``tests/test_service_parity.py``) asserts exactly that
against the reference functions for every ``kettle_config.yaml`` label and
a rel-type sample.

Pure stdlib + PyYAML. No torch / neo4j import here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_YAML = Path(__file__).with_name("schema_map.yaml")


@dataclass(frozen=True)
class TemporalSpec:
    year_label: str
    year_property: str
    max_hops: int


@dataclass(frozen=True)
class _EdgeRule:
    category: int
    contains: tuple[str, ...]


class SchemaMap:
    """Loaded ``schema_map.yaml``. Maps a node's primary label -> KGR layer
    id and a rel-type -> KGR edge-category id, reproducing the hardcoded
    ``neo4j_eval_export`` reference exactly.

    Construct with :meth:`from_yaml` (defaults to the sibling
    ``schema_map.yaml``).
    """

    def __init__(self, data: dict[str, Any], source: str | Path | None = None):
        self.source = str(source) if source is not None else "<dict>"
        nl = data.get("node_layers", {}) or {}
        self._layer_default = int(nl.get("default", 2))
        self._aux_layer = int(nl.get("aux_layer", 3))
        self._source_layer = int(nl.get("source_layer", 0))
        self._claim_layer = int(nl.get("claim_layer", 1))
        # exact-label sets (primary label), case-sensitive (mirrors the
        # reference, which compares the raw Neo4j label string).
        self._aux_labels = frozenset(nl.get("aux_labels", []) or [])
        self._source_labels = frozenset(nl.get("source_labels", []) or [])
        self._claim_labels = frozenset(nl.get("claim_labels", []) or [])

        ec = data.get("edge_categories", {}) or {}
        self._cat_default = int(ec.get("default", 2))
        rules: list[_EdgeRule] = []
        for r in ec.get("rules", []) or []:
            rules.append(
                _EdgeRule(
                    category=int(r["category"]),
                    # uppercased once here; matched as substrings, which is
                    # exactly `any(k in rel.upper() for k in contains)`.
                    contains=tuple(str(s).upper() for s in r["contains"]),
                )
            )
        self._edge_rules = tuple(rules)

        t = data.get("temporal", {}) or {}
        self.temporal = TemporalSpec(
            year_label=str(t.get("year_label", "Year")),
            year_property=str(t.get("year_property", "year")),
            max_hops=int(t.get("max_hops", 3)),
        )
        self.identity_seed_property = (
            (data.get("identity", {}) or {}).get("seed_property")
        )
        self.version = int(data.get("version", 1))

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "SchemaMap":
        p = Path(path) if path is not None else _DEFAULT_YAML
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data, source=p)

    # -- the two mappings ----------------------------------------------------

    def is_aux(self, primary_label: str) -> bool:
        """Mirror of ``neo4j_eval_export._AUX_LABELS`` membership."""
        return primary_label in self._aux_labels

    def node_layer(self, primary_label: str) -> int:
        """Mirror of ``neo4j_eval_export._node_layer`` (line 144-151),
        including its precedence: auxiliary, then source (provenance
        roots), then claim (intermediate assertions), else the entity
        (default) layer. Yamls without source/claim label sets reproduce
        the pre-4-way binary mapping unchanged."""
        if primary_label in self._aux_labels:
            return self._aux_layer
        if primary_label in self._source_labels:
            return self._source_layer
        if primary_label in self._claim_labels:
            return self._claim_layer
        return self._layer_default

    def edge_category(self, rel_type: str) -> int:
        """Mirror of ``neo4j_eval_export._edge_category`` (line 129-141):
        the first ordered rule any of whose substrings appears in the
        uppercased rel-type wins; otherwise the structural default.

        Substring containment + rule order reproduce the original
        ``any(k in r for k in (...))`` if/elif chain bit-for-bit."""
        r = rel_type.upper()
        for rule in self._edge_rules:
            if any(k in r for k in rule.contains):
                return rule.category
        return self._cat_default


# ---------------------------------------------------------------------------
# P0 golden no-op self-check (also importable by verify.py / the pytest)
# ---------------------------------------------------------------------------

def golden_noop_check(
    schema_map: SchemaMap,
    labels: list[str],
    rel_types: list[str],
) -> dict:
    """Assert ``schema_map`` reproduces the hardcoded reference for every
    label/rel-type. Returns a report dict; ``ok`` is True iff bit-exact.

    The reference is the live ``neo4j_eval_export`` module (the single
    source of truth being de-hardcoded), imported with ``scripts/`` on the
    path -- the same path shim ``smoke_retrieval_workflow.py`` uses.
    """
    import sys

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import neo4j_eval_export as ref  # type: ignore

    layer_mismatch = [
        {"label": L, "schema_map": schema_map.node_layer(L),
         "reference": ref._node_layer(L)}
        for L in labels
        if schema_map.node_layer(L) != ref._node_layer(L)
    ]
    cat_mismatch = [
        {"rel_type": rt, "schema_map": schema_map.edge_category(rt),
         "reference": ref._edge_category(rt)}
        for rt in rel_types
        if schema_map.edge_category(rt) != ref._edge_category(rt)
    ]
    return {
        "ok": not layer_mismatch and not cat_mismatch,
        "n_labels": len(labels),
        "n_rel_types": len(rel_types),
        "layer_mismatch": layer_mismatch,
        "category_mismatch": cat_mismatch,
        "source": schema_map.source,
    }


if __name__ == "__main__":
    # Quick standalone P0 over the kettle_config labels + a rel-type probe.
    import json
    import sys

    sm = SchemaMap.from_yaml()
    cfg_path = Path(__file__).resolve().parents[2] / "scripts" / "kettle_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    labels = sorted(set(cfg.get("required_properties", {}).keys()))
    rel_probe = [
        "SOURCED_FROM", "DERIVED_FROM", "PROVENANCE_OF", "EVIDENCED_BY",
        "REFERS_TO", "MENTIONS", "ABOUT", "DESCRIBES", "IN_YEAR",
        "HAS_PERIOD", "TEMPORAL_SCOPE", "DATED", "SUPERSEDES",
        "CORROBORATES", "SCOPED_TO", "CO_OCCURS_WITH", "RELATED_TO",
        "ASSOCIATED_WITH", "SAME_AS", "LINKED_TO", "NEAR", "OCCURRED_AT",
        "AT_PLACE", "HAS_HABITAT", "OBSERVED", "PART_OF",
    ]
    rep = golden_noop_check(sm, labels, rel_probe)
    print(json.dumps(rep, indent=2))
    sys.exit(0 if rep["ok"] else 1)
