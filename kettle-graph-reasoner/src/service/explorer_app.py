r"""KGR Neo4j Explorer -- a local web app to interact with the live graph
through the KGR context model.

A published claude.ai artifact CANNOT reach Neo4j (sandbox CSP blocks all
network; the model needs Python/torch). So this is a LOCAL app: it serves
a UI on localhost, talks to Neo4j with the project driver, and ranks live
neighborhoods with the frozen KGR encoder.

    py -m src.service.explorer_app          # then open http://127.0.0.1:8765

Two panels:
  - Cypher browser (READ-ONLY): run a query, see rows / node cards.
  - KGR context: load a live neighborhood once, then click any node to
    re-anchor and re-rank instantly (multi-select for multi-anchor).

Read-only is enforced: write clauses are rejected before hitting the DB.
"""

from __future__ import annotations

import re
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, Response

from .context_service import KGRContextService

# lazy singletons
_SVC = None
_DRIVER = None
_BUNDLE = None                     # domain-only graphcache + arrays (heavy; built once)
_NB: dict[str, dict] = {}          # token -> loaded neighborhood
DOMAIN_LABELS = ["Entity", "Person", "Place", "Organization", "Refuge",
                 "Species", "Habitat", "Parcel", "Event", "Activity",
                 "Observation", "Measurement", "Period", "Year", "Concept",
                 "SurveyMethod"]
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # kettle-graph-reasoner/
WRITE_RE = re.compile(
    r"\b(create|merge|delete|set|remove|drop|foreach|call\s*\{[^}]*\b(create|delete|set)\b)",
    re.IGNORECASE)

app = Flask(__name__)


def svc() -> KGRContextService:
    global _SVC
    if _SVC is None:
        _SVC = KGRContextService()
    return _SVC


def driver():
    global _DRIVER
    if _DRIVER is None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import neo4j_reader  # noqa: E402
        _DRIVER = neo4j_reader.get_driver()
    return _DRIVER


def read_session():
    """Server-enforced read-only session. Every explorer query runs through
    this: transactions are marked READ, so write statements are rejected by
    the SERVER (Neo.ClientError.Statement.AccessMode), not by pattern
    matching. The WRITE_RE denylist on /api/cypher stays only as a
    friendlier fast-path error; it is NOT the security boundary. For a
    privileged deployment the outer boundary must still be read-only
    database credentials -- admin procedures allowed to the *user* are not
    blocked by access mode."""
    try:
        from neo4j import READ_ACCESS  # noqa: E402
    except ImportError:                 # very old driver: string constant
        READ_ACCESS = "READ"
    return driver().session(default_access_mode=READ_ACCESS)


def _fetch_nodes(ids: list[int]) -> dict[int, dict]:
    """labels + a display name + a few props for a set of legacy ids."""
    if not ids:
        return {}
    # prefer a DESCRIPTIVE property over the year, so same-era nodes are
    # distinguishable (Events fall back to event_type, not "1903").
    q = ("MATCH (n) WHERE id(n) IN $ids "
         "RETURN id(n) AS id, labels(n) AS labels, "
         "coalesce(n.name, n.title, n.value, n.text, n.event_type, "
         "         n.observation_type, n.activity_type, n.entity_type, "
         "         n.year, n.paragraph_id, n.doc_id, toString(id(n))) AS name, "
         "properties(n) AS props")
    out = {}
    with read_session() as s:
        for r in s.run(q, ids=[int(i) for i in ids]):
            props = dict(r["props"])
            props.pop("embedding", None)
            small = {k: v for k, v in list(props.items())[:6]
                     if not isinstance(v, (list, dict))}
            out[int(r["id"])] = {
                "id": int(r["id"]),
                "labels": list(r["labels"]),
                "name": str(r["name"])[:80],
                "props": small,
            }
    return out


def _exporter():
    """Import the exporter module (puts scripts/ + repo on sys.path)."""
    driver()  # ensures scripts/ on sys.path
    import neo4j_eval_export as X  # noqa: E402
    return X


def _domain_bundle():
    """Build the domain-only graphcache + derived arrays ONCE (seconds).
    Mirrors neo4j_eval_export.cmd_export setup so anchor-centered
    neighborhoods encode identically to the sampled corpus."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    X = _exporter()
    from graph_diagnostics.core import lifecycle_predicate
    from graph_diagnostics.graphcache import build_cache, connected_components
    cfg, spec = X._spec_from_config(str(ROOT / "scripts" / "kettle_config.yaml"))
    with read_session() as s:
        cache = build_cache(s, lifecycle_predicate(cfg, var="n"))
        node_mask = cache.label_mask(spec.get("include_labels") or [])
        edge_mask = cache.edge_mask(node_mask,
                                    spec.get("include_rel_types") or [],
                                    spec.get("exclude_rel_types") or [])
        indptr, indices = cache.induced_csr(edge_mask)
        # broader CSR (all domain edge types) for growing anchor-centred
        # balls -- the spec's rel-type filter is too sparse for arbitrary
        # anchors (e.g. an Organization). Encoding still pulls every edge
        # among the chosen nodes, so this only affects which nodes are picked.
        indptr_all, indices_all = cache.induced_csr(
            cache.edge_mask(node_mask, [], []))
        yvals = X._pull_year_values(s, cache.id2idx)
        min_y, max_y, has_y = cache.ensure_year_reachability(
            "Year", yvals, cfg.temporal_max_hops)
        lo, hi = ((min(yvals.values()), max(yvals.values()))
                  if yvals else (0.0, 1.0))
        span = (hi - lo) or 1.0
        t_start = np.clip(np.where(has_y, (min_y - lo) / span, 0.0),
                          0, 1).astype(np.float64)
        t_end = np.clip(np.where(has_y, (max_y - lo) / span, 0.0),
                        0, 1).astype(np.float64)
        comp = connected_components(indptr, indices, cache.n, allowed=node_mask)
        lab = comp[node_mask]
        giant = int(np.bincount(lab[lab >= 0]).argmax())
        giant_member = node_mask & (comp == giant)
        idx2id = np.empty(cache.n, dtype=np.int64)
        for nid, ix in cache.id2idx.items():
            idx2id[ix] = nid
    _BUNDLE = {"cache": cache, "indptr": indptr, "indices": indices,
               "indptr_all": indptr_all, "indices_all": indices_all,
               "giant_member": giant_member, "node_mask": node_mask,
               "idx2id": idx2id, "t_start": t_start, "t_end": t_end}
    return _BUNDLE


@app.post("/api/search")
def search():
    q = (request.json or {}).get("q", "").strip()
    if len(q) < 2 and not q.isdigit():
        return jsonify({"error": "type at least 2 characters"}), 400
    name = ("coalesce(n.name, n.title, n.event_type, n.observation_type, "
            "n.activity_type, n.year, toString(id(n)))")
    try:
        with read_session() as s:
            if q.isdigit():
                cy = ("MATCH (n) WHERE id(n) = $nid "
                      f"RETURN id(n) AS id, labels(n) AS labels, {name} AS name, "
                      "properties(n) AS props")
                res = s.run(cy, nid=int(q))
            else:
                cy = ("MATCH (n) WHERE any(l IN labels(n) WHERE l IN $dom) AND ("
                      "toLower(coalesce(n.name,n.title,n.event_type,"
                      "n.observation_type,n.activity_type,n.value,n.text,'')) "
                      "CONTAINS $q OR any(l IN labels(n) WHERE toLower(l) "
                      "CONTAINS $q)) "
                      f"RETURN id(n) AS id, labels(n) AS labels, {name} AS name, "
                      "properties(n) AS props LIMIT 25")
                res = s.run(cy, q=q.lower(), dom=DOMAIN_LABELS)
            out = []
            for r in res:
                props = {k: v for k, v in list(dict(r["props"]).items())[:4]
                         if not isinstance(v, (list, dict))}
                out.append({"id": int(r["id"]), "labels": list(r["labels"]),
                            "name": str(r["name"])[:60], "props": props})
            return jsonify({"results": out})
    except Exception as ex:
        return jsonify({"error": f"{type(ex).__name__}: {ex}"}), 400


def _store_neighborhood(z, focus_id=None):
    """Embed an npz neighborhood dict, keep it, return the client payload."""
    import uuid as _uuid
    h = svc().load_graph(z)
    ids = [int(x) for x in h.node_ids]
    id2row = {nid: r for r, nid in enumerate(ids)}
    meta = _fetch_nodes(ids)
    if focus_id is not None and focus_id in id2row:
        anchor_id = focus_id
    else:
        anchor_id = ids[int(z["task_0_anchor_row"])]
    token = _uuid.uuid4().hex[:10]
    _NB[token] = {"handle": h, "ids": ids, "id2row": id2row,
                  "meta": meta, "anchor_id": anchor_id}
    nodes = [{"id": nid, **meta.get(nid, {"labels": [], "name": str(nid),
                                          "props": {}})} for nid in ids]
    return {"token": token, "n": h.n, "nodes": nodes, "anchor_id": anchor_id}


@app.post("/api/load_anchor")
def load_anchor():
    body = request.json or {}
    node_id = int(body.get("node_id"))
    max_nodes = int(body.get("max_nodes", 250))
    try:
        b = _domain_bundle()
    except Exception as ex:
        return jsonify({"error": f"graphcache build failed: {ex}"}), 500
    seed_idx = b["cache"].id2idx.get(node_id)
    member = b["node_mask"]
    if seed_idx is None or not bool(member[seed_idx]):
        return jsonify({"error": "that node isn't in the domain layer â€” pick a "
                        "domain entity (Event, Observation, Period, Person, "
                        "Place, Organization, ...)."}), 400
    X = _exporter()
    nodes = X._bfs_ball(b["indptr_all"], b["indices_all"], member,
                        int(seed_idx), max_nodes)
    if nodes.size < 8:
        return jsonify({"error": "that node has too few domain neighbors to "
                        "form a neighborhood (try a more connected entity, "
                        "e.g. an Event or Place)."}), 400
    try:
        with read_session() as s:
            npz = X._encode_graph(s, b["cache"], nodes, int(seed_idx),
                                  b["idx2id"], b["t_start"], b["t_end"],
                                  np.random.default_rng(0), 1, 0)
    except Exception as ex:
        return jsonify({"error": f"encode failed: {type(ex).__name__}: {ex}"}), 500
    return jsonify(_store_neighborhood(npz, focus_id=node_id))


@app.get("/")
def index() -> Response:
    return Response((HERE / "explorer.html").read_text(encoding="utf-8"),
                    mimetype="text/html")


@app.get("/api/health")
def health():
    info = {"model": True, "neo4j": False, "db": {}}
    try:
        with read_session() as s:
            n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            e = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            info["neo4j"] = True
            info["db"] = {"nodes": int(n), "rels": int(e)}
    except Exception as ex:
        info["error"] = f"{type(ex).__name__}: {ex}"
    return jsonify(info)


@app.post("/api/cypher")
def cypher():
    q = (request.json or {}).get("cypher", "").strip()
    if not q:
        return jsonify({"error": "empty query"}), 400
    if WRITE_RE.search(q):
        return jsonify({"error": "read-only: write clauses are blocked "
                        "(create/merge/delete/set/remove/drop)."}), 400
    try:
        with read_session() as s:
            res = s.run(q)
            cols = res.keys()
            rows = []
            for rec in res:
                row = {}
                for k in cols:
                    v = rec[k]
                    if hasattr(v, "labels"):        # a Node
                        v = {"_node": True, "id": v.id,
                             "labels": list(v.labels),
                             "props": {kk: vv for kk, vv in dict(v).items()
                                       if not isinstance(vv, (list, dict))}}
                    elif isinstance(v, (list, dict)):
                        v = str(v)[:200]
                    row[k] = v
                rows.append(row)
                if len(rows) >= 200:
                    break
            return jsonify({"columns": list(cols), "rows": rows,
                            "truncated": len(rows) >= 200})
    except Exception as ex:
        return jsonify({"error": f"{type(ex).__name__}: {ex}"}), 400


@app.post("/api/load")
def load_neighborhood():
    """Export ONE live neighborhood, embed it, keep it in memory."""
    body = request.json or {}
    seed = int(body.get("seed", 42))
    max_nodes = int(body.get("max_nodes", 300))
    token = uuid.uuid4().hex[:10]
    out = ROOT / "runs" / "_explorer" / token
    out.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [sys.executable, "neo4j_eval_export.py", "export",
             "--config", "kettle_config.yaml", "--out", str(out),
             "--num-graphs", "1", "--max-nodes", str(max_nodes),
             "--tasks-per-graph", "1", "--seed", str(seed),
             "--sampler", "delocalized", "--n-seeds", "4"],
            cwd=str(ROOT / "scripts"), check=True, capture_output=True,
            timeout=180)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "export timed out (Neo4j slow/unreachable)"}), 504
    except subprocess.CalledProcessError as ex:
        msg = (ex.stderr or b"").decode("utf-8", "ignore")[-400:]
        return jsonify({"error": f"export failed: {msg}"}), 500
    files = sorted(out.glob("graph_*.npz"))
    if not files:
        return jsonify({"error": "no neighborhood produced"}), 500
    z = dict(np.load(files[0], allow_pickle=True))
    return jsonify(_store_neighborhood(z))


@app.post("/api/reorder")
def reorder():
    """Re-rank a loaded neighborhood by one or more anchor node ids (fast)."""
    body = request.json or {}
    token = body.get("token")
    nb = _NB.get(token)
    if not nb:
        return jsonify({"error": "neighborhood not loaded (re-load)"}), 400
    anchor_ids = body.get("anchors") or [nb["anchor_id"]]
    top_k = int(body.get("top_k", 15))
    ball = body.get("ball_hops")
    rows = [nb["id2row"][int(a)] for a in anchor_ids if int(a) in nb["id2row"]]
    if not rows:
        return jsonify({"error": "anchor(s) not in this neighborhood"}), 400
    res = svc().order_context(nb["handle"], rows, top_k=top_k,
                              ball_hops=(int(ball) if ball else None))
    items = []
    for it in res.items:
        nid = int(it.node_id)
        m = nb["meta"].get(nid, {"labels": [], "name": str(nid), "props": {}})
        items.append({"rank": it.rank, "id": nid, "score": round(it.score, 4),
                      "hop": it.hop, "rationale": it.rationale,
                      "labels": m["labels"], "name": m["name"],
                      "props": m["props"]})
    return jsonify({"anchors": [int(a) for a in anchor_ids],
                    "n_candidates": res.n_candidates,
                    "spread": round(res.discrimination, 4), "items": items})


def _edges_among(ids: list[int]) -> list[tuple]:
    """Directed relations among a node set (the subgraph structure)."""
    if len(ids) < 2:
        return []
    q = ("MATCH (a)-[r]->(b) WHERE id(a) IN $ids AND id(b) IN $ids "
         "RETURN id(a) AS a, type(r) AS t, id(b) AS b")
    with read_session() as s:
        return [(int(r["a"]), r["t"], int(r["b"])) for r in s.run(q, ids=ids)]


def _serialize_context(nb, rows: list[int], anchor_rows: list[int]) -> str:
    """Render anchor + selected rows as a NODES + RELATIONS block for the LLM.
    Anchors are included and marked so the model can tie context to them."""
    meta = nb["meta"]
    aset = {nb["ids"][r] for r in anchor_rows}
    all_rows = list(dict.fromkeys(list(anchor_rows) + list(rows)))
    ids = [nb["ids"][r] for r in all_rows]
    lines = ["NODES:"]
    for nid in ids:
        m = meta.get(nid, {"labels": [], "name": str(nid), "props": {}})
        props = " | ".join(f"{k}={v}" for k, v in list(m["props"].items())[:5]
                           if k not in ("run_id",))
        lab = ":".join(m["labels"][:2]) or "?"
        tag = " (ANCHOR)" if nid in aset else ""
        lines.append(f"  #{nid}{tag} [{lab}] {m['name']}"
                     + (f" | {props}" if props else ""))
    edges = _edges_among(ids)
    if edges:
        nm = {nid: (meta.get(nid, {}).get("name") or str(nid)) for nid in ids}
        lines.append("RELATIONS:")
        for a, t, b in edges[:80]:
            lines.append(f"  #{a} ({nm.get(a,'')}) -{t}-> #{b} ({nm.get(b,'')})")
    return "\n".join(lines)


@app.get("/api/llm")
def llm_info():
    from . import llm
    st = llm.discover()
    return jsonify({"available": bool(st.get("base")), "base": st.get("base"),
                    "models": st.get("models", []), "default": st.get("default")})


@app.post("/api/ask")
def ask():
    from . import llm
    body = request.json or {}
    token = body.get("token")
    nb = _NB.get(token)
    if not nb:
        return jsonify({"error": "load a neighborhood first"}), 400
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400
    mode = body.get("mode", "kgr")          # kgr | bfs | none
    anchor_ids = body.get("anchors") or [nb["anchor_id"]]
    top_k = int(body.get("top_k", 12))
    model = body.get("model")
    # Same contract as /api/reorder: invalid anchors are a caller error.
    # NEVER silently substitute a fallback row — the prompt names the
    # caller's anchors, so a substituted anchor produces confidently wrong
    # context (the anchor-fragility failure mode; see suggester findings).
    anchor_rows = [nb["id2row"][int(a)] for a in anchor_ids
                   if int(a) in nb["id2row"]]
    if not anchor_rows:
        return jsonify({"error": "anchor(s) not in this neighborhood: "
                        f"{anchor_ids}"}), 400
    dropped = [int(a) for a in anchor_ids if int(a) not in nb["id2row"]]
    if dropped:  # partial: keep valid ones, but never lie in the prompt
        anchor_ids = [int(a) for a in anchor_ids if int(a) in nb["id2row"]]

    # ---- select the context nodes ----
    if mode == "none":
        rows = []
    elif mode == "bfs":
        mh = nb["handle"].min_hops(anchor_rows)
        order = sorted((r for r in range(nb["handle"].n) if r not in anchor_rows),
                       key=lambda r: (mh[r] if mh[r] >= 0 else 10**9,
                                      str(nb["ids"][r])))
        rows = order[:top_k]
    else:  # kgr
        res = svc().order_context(nb["handle"], anchor_rows, top_k=top_k)
        rows = [it.row for it in res.items]

    ctx = (_serialize_context(nb, rows, anchor_rows) if rows
           else "(no context provided)")
    anchor_desc = ", ".join(
        f"#{a} {nb['meta'].get(a,{}).get('name','')}" for a in anchor_ids)
    system = (
        "You answer questions about the Turnbull archival knowledge graph. "
        "Use ONLY the CONTEXT below -- a subgraph selected as most relevant to "
        "the question's anchor entities. Cite node #ids you rely on. If the "
        "context does not contain the answer, say so plainly. Be concise.")
    user = (f"ANCHOR ENTITIES: {anchor_desc}\n\nCONTEXT:\n{ctx}\n\n"
            f"QUESTION: {question}")
    try:
        out = llm.chat([{"role": "system", "content": system},
                        {"role": "user", "content": user}], model=model)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 502
    return jsonify({"answer": out["text"], "model": out["model"],
                    "mode": mode, "n_context_nodes": len(rows),
                    "anchors_used": [int(a) for a in anchor_ids],
                    "anchors_dropped": dropped,
                    "context": ctx, "usage": out.get("usage", {})})


def main() -> None:
    port = 8765
    print("=" * 58)
    print("  KGR Neo4j Explorer")
    print(f"  open  ->  http://127.0.0.1:{port}")
    print("  (loads the frozen model + Neo4j driver on first request)")
    print("=" * 58)
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
