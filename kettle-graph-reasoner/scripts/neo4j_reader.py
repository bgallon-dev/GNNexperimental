"""
neo4j_reader.py
===============

Read-side Neo4j utility for the Kettle stack.

Two entry points:

1. run_query(name, **params) -> pandas.DataFrame
       Named-query library. Register queries in the QUERIES dict and invoke
       by name. Results are materialized to a DataFrame -- appropriate for
       small-to-medium result sets (ad-hoc analysis, notebook work,
       entity-resolution inspection, IHT encounter queries).

2. stream_export(query, output_path, format, **params) -> int
       Streams records one-at-a-time to JSONL, CSV, or Parquet. Never
       materializes the full result set in memory. Use for large subgraph
       exports (e.g., the Spokane corpus entity graph, KGR training data).
       Returns the number of records written.

Environment
-----------
Expects a `.env` file in the same directory (or any parent discoverable by
python-dotenv) with at minimum:

    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your-password

Dependencies
------------
    pip install neo4j python-dotenv pandas pyarrow
"""

from __future__ import annotations

import csv
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from neo4j import Driver, GraphDatabase, Record, Session


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Walk upward from this script to find the nearest .env (project root sits
# two levels above scripts/). Falls back to CWD search if none is found.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ENV_PATH = find_dotenv(filename=".env", usecwd=False, raise_error_if_not_found=False)
if not _ENV_PATH:
    for _parent in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
        _candidate = _parent / ".env"
        if _candidate.is_file():
            _ENV_PATH = str(_candidate)
            break
load_dotenv(_ENV_PATH or None)

ExportFormat = Literal["jsonl", "csv", "parquet"]


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {key}. "
            f"Check your .env file at {_SCRIPT_DIR / '.env'}."
        )
    return val


# ---------------------------------------------------------------------------
# Named query library
# ---------------------------------------------------------------------------
#
# Register Cypher queries here. Keys are short identifiers; values are the
# Cypher strings. Use $param placeholders -- parameters are passed through
# kwargs on run_query() / stream_export().

QUERIES: dict[str, str] = {
    "node_count_by_label": """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC
    """,
    "relationship_count_by_type": """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(*) AS count
        ORDER BY count DESC
    """,
    "sample_nodes_by_label": """
        MATCH (n)
        WHERE $label IN labels(n)
        RETURN n
        LIMIT $limit
    """,
    # Add project-specific queries here, e.g. IHT encounter queries,
    # entity-resolution spot-checks, KGR subgraph pulls, etc.
}


# ---------------------------------------------------------------------------
# Driver / session management
# ---------------------------------------------------------------------------

_driver: Driver | None = None


def get_driver() -> Driver:
    """Lazily construct and cache a singleton Driver.

    The driver is thread-safe and intended to be long-lived per process.
    Call close_driver() at shutdown.
    """
    global _driver
    if _driver is None:
        uri = _require_env("NEO4J_URI")
        username = _require_env("NEO4J_USERNAME")
        password = _require_env("NEO4J_PASSWORD")
        _driver = GraphDatabase.driver(uri, auth=(username, password))
        # Fail fast if credentials are wrong or the server is unreachable.
        _driver.verify_connectivity()
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


@contextmanager
def session() -> Iterator[Session]:
    """Yield a Session scoped to a single logical unit of work."""
    driver = get_driver()
    with driver.session() as s:
        yield s


# ---------------------------------------------------------------------------
# Entry point 1: named-query library -> DataFrame
# ---------------------------------------------------------------------------

def run_query(name: str, **params: Any) -> pd.DataFrame:
    """Execute a named query and return results as a DataFrame.

    Appropriate for result sets that fit comfortably in memory. For large
    exports use stream_export() instead.

    Parameters
    ----------
    name : str
        Key into QUERIES.
    **params
        Cypher parameters (e.g. label="Person", limit=100).

    Returns
    -------
    pandas.DataFrame
        One row per record. Nested nodes/relationships are kept as dicts.
    """
    if name not in QUERIES:
        raise KeyError(
            f"Unknown query {name!r}. Registered queries: {sorted(QUERIES)}"
        )
    cypher = QUERIES[name]
    with session() as s:
        result = s.run(cypher, **params)  # pyright: ignore[reportArgumentType]
        rows = [_record_to_dict(r) for r in result]
    return pd.DataFrame(rows)


def run_cypher(cypher: str, **params: Any) -> pd.DataFrame:
    """Execute an ad-hoc Cypher string (escape hatch for exploratory work)."""
    with session() as s:
        result = s.run(cypher, **params)  # pyright: ignore[reportArgumentType]
        rows = [_record_to_dict(r) for r in result]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point 2: streaming export
# ---------------------------------------------------------------------------

def stream_export(
    query: str,
    output_path: str | Path,
    format: ExportFormat = "jsonl",
    batch_size: int = 10_000,
    **params: Any,
) -> int:
    """Stream a query result to disk without materializing it in memory.

    Parameters
    ----------
    query : str
        Either a key in QUERIES or a raw Cypher string. If the string
        contains whitespace it is treated as raw Cypher.
    output_path : str | Path
        Destination file. Extension is NOT auto-appended -- pass it.
    format : {"jsonl", "csv", "parquet"}
        Output format. JSONL preserves nested structure (nodes as dicts);
        CSV flattens to string columns; Parquet batches via pyarrow.
    batch_size : int
        Records per batch for Parquet writes. Ignored for JSONL/CSV.
    **params
        Cypher parameters.

    Returns
    -------
    int
        Number of records written.
    """
    cypher = QUERIES[query] if query in QUERIES else query
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with session() as s:
        result = s.run(cypher, **params)  # pyright: ignore[reportArgumentType]

        if format == "jsonl":
            return _stream_jsonl(result, output_path)
        if format == "csv":
            return _stream_csv(result, output_path)
        if format == "parquet":
            return _stream_parquet(result, output_path, batch_size)
        raise ValueError(f"Unknown format: {format!r}")


# ---------------------------------------------------------------------------
# Format-specific streaming writers
# ---------------------------------------------------------------------------

def _stream_jsonl(result, output_path: Path) -> int:
    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in result:
            f.write(json.dumps(_record_to_dict(record), default=_json_default))
            f.write("\n")
            n += 1
    return n


def _stream_csv(result, output_path: Path) -> int:
    n = 0
    writer: csv.DictWriter | None = None
    with output_path.open("w", encoding="utf-8", newline="") as f:
        for record in result:
            row = _record_to_dict(record, flatten=True)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            n += 1
    return n


def _stream_parquet(result, output_path: Path, batch_size: int) -> int:
    # Imported lazily so users without pyarrow aren't blocked.
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 0
    batch: list[dict] = []
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None

    def flush():
        nonlocal writer, schema
        if not batch:
            return
        table = pa.Table.from_pylist(batch)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(output_path, schema)
        writer.write_table(table)
        batch.clear()

    try:
        for record in result:
            batch.append(_record_to_dict(record, flatten=True))
            n += 1
            if len(batch) >= batch_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    return n


# ---------------------------------------------------------------------------
# Record conversion
# ---------------------------------------------------------------------------

def _record_to_dict(record: Record, flatten: bool = False) -> dict[str, Any]:
    """Convert a neo4j Record into a plain dict.

    When flatten=True, Node/Relationship values are rendered as JSON strings
    so they fit cleanly into CSV/Parquet columns.
    """
    out: dict[str, Any] = {}
    for key, value in record.items():
        converted = _convert_value(value)
        if flatten and isinstance(converted, (dict, list)):
            out[key] = json.dumps(converted, default=_json_default)
        else:
            out[key] = converted
    return out


def _convert_value(value: Any) -> Any:
    # neo4j.graph types expose .items() for properties; detect duck-typed.
    if hasattr(value, "labels") and hasattr(value, "items"):
        return {
            "_type": "node",
            "_labels": list(value.labels),
            "_id": value.element_id,
            **dict(value.items()),
        }
    if hasattr(value, "type") and hasattr(value, "start_node"):
        return {
            "_type": "relationship",
            "_rel_type": value.type,
            "_id": value.element_id,
            "_start": value.start_node.element_id,
            "_end": value.end_node.element_id,
            **dict(value.items()),
        }
    if hasattr(value, "nodes") and hasattr(value, "relationships"):
        return {
            "_type": "path",
            "nodes": [_convert_value(n) for n in value.nodes],
            "relationships": [_convert_value(r) for r in value.relationships],
        }
    if isinstance(value, (list, tuple)):
        return [_convert_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    return value


def _json_default(obj: Any) -> Any:
    # neo4j temporal types and anything else exotic.
    return str(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j reader (Kettle stack)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List registered named queries")
    p_list.set_defaults(func=lambda args: _cmd_list())

    p_run = sub.add_parser("run", help="Run a named query and print results")
    p_run.add_argument("name")
    p_run.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Cypher parameter (repeatable). Values parsed as JSON if possible.",
    )
    p_run.set_defaults(func=_cmd_run)

    p_export = sub.add_parser("export", help="Stream a query to disk")
    p_export.add_argument("query", help="Named query key or raw Cypher")
    p_export.add_argument("output", help="Output file path")
    p_export.add_argument(
        "--format", choices=("jsonl", "csv", "parquet"), default="jsonl"
    )
    p_export.add_argument("--batch-size", type=int, default=10_000)
    p_export.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    p_export.set_defaults(func=_cmd_export)

    args = parser.parse_args()
    try:
        return args.func(args)
    finally:
        close_driver()


def _parse_params(pairs: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Bad --param {pair!r}; expected KEY=VALUE")
        k, v = pair.split("=", 1)
        try:
            params[k] = json.loads(v)
        except json.JSONDecodeError:
            params[k] = v
    return params


def _cmd_list() -> int:
    for name in sorted(QUERIES):
        print(name)
    return 0


def _cmd_run(args) -> int:
    df = run_query(args.name, **_parse_params(args.param))
    # Print head + summary; full DataFrame can be huge.
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(50).to_string(index=False))
    print(f"\n[{len(df)} rows x {len(df.columns)} cols]", file=sys.stderr)
    return 0


def _cmd_export(args) -> int:
    n = stream_export(
        args.query,
        args.output,
        format=args.format,
        batch_size=args.batch_size,
        **_parse_params(args.param),
    )
    print(f"Wrote {n} records to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
