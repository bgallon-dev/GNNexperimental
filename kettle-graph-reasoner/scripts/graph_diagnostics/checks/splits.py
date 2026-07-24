"""
Train/val/test split feasibility.

What counts as a valid split depends on the task:

    link_prediction:
        - Need enough edges (default: >= 1000 positives).
        - Each class of relationship type should appear in all three splits.
        - After removing test edges, the remaining graph should still be
          (near-)connected or the supervision becomes degenerate.
        - Negative sampling should be feasible -- i.e. the graph is not so
          dense that non-edges are pathological (here we flag density > 0.2).

    node_classification:
        - Need a label property whose values we can enumerate.
        - Each class must appear N>=10 times to survive an 80/10/10 split.
        - Class imbalance > 100:1 warrants stratified sampling.

    entity_res:
        - Need candidate-pair positives (e.g. nodes linked by a SAME_AS rel,
          or groups from the entity_res check). Report pair count and
          suggest stratification by block.

For multi-task (your KGR case), we run all three and report per-task.
"""
from __future__ import annotations

from collections import Counter

from graph_diagnostics.core import (
    CheckResult, Finding, Severity, DiagnosticConfig,
)


def run(session, config: DiagnosticConfig) -> CheckResult:
    result = CheckResult(check="splits")

    tasks = [config.split_task] if config.split_task != "multi" else [
        "link_prediction", "node_classification", "entity_res",
    ]
    for task in tasks:
        if task == "link_prediction":
            _check_link_prediction(session, config, result)
        elif task == "node_classification":
            _check_node_classification(session, config, result)
        elif task == "entity_res":
            _check_entity_res_splits(session, config, result)
        elif task == "er_mention_entity":
            _check_er_mention_entity(session, config, result)

    return result


def _check_link_prediction(session, config, result) -> None:
    row = session.run("""
        MATCH (n) WITH count(n) AS nodes
        OPTIONAL MATCH ()-[r]->() WITH nodes, count(r) AS edges
        RETURN nodes, edges
    """).single()
    n, m = row["nodes"], row["edges"] or 0

    tr, va, te = config.split_ratios
    test_edges = int(m * te)
    val_edges = int(m * va)

    if m < 1000:
        result.findings.append(Finding(
            check="splits", code="lp_too_few_edges",
            severity=Severity.HIGH,
            message=(
                f"Only {m} edges; link prediction typically needs >= 1000 for "
                f"a meaningful {int(te*100)}% test split (would yield {test_edges})."
            ),
            count=m,
        ))

    density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0
    if density > 0.2:
        result.findings.append(Finding(
            check="splits", code="lp_dense_negative_sampling",
            severity=Severity.MEDIUM,
            message=(
                f"Graph density {density:.3f} is high for negative sampling. "
                f"Random non-edges will increasingly coincide with true "
                f"missing edges. Consider hard-negative mining."
            ),
            count=m,
        ))

    # Per-relationship-type coverage.
    rel_counts = Counter()
    for row in session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c"):
        rel_counts[row["t"]] = row["c"]
    rare = [(t, c) for t, c in rel_counts.items() if c < 10]
    if rare:
        result.findings.append(Finding(
            check="splits", code="lp_rare_rel_types",
            severity=Severity.MEDIUM,
            message=(
                f"{len(rare)} relationship types have < 10 edges; these cannot "
                f"be stratified across train/val/test and should be merged into "
                f"a generic type or excluded from evaluation."
            ),
            count=len(rare),
            details={"rare_types": dict(rare)},
        ))

    result.findings.append(Finding(
        check="splits", code="lp_plan",
        severity=Severity.INFO,
        message=(
            f"Link-prediction split plan ({int(tr*100)}/{int(va*100)}/{int(te*100)}): "
            f"~{m - val_edges - test_edges} train / {val_edges} val / {test_edges} test."
        ),
        count=m,
    ))


def _check_node_classification(session, config, result) -> None:
    # We need a label property. Look for a 'class' or 'category' property on
    # any label; if the user hasn't configured, just report what we find.
    candidate_props = ["class", "category", "type", "kind", "label"]
    found = []
    for label_row in session.run("CALL db.labels() YIELD label RETURN label"):
        label = label_row["label"]
        for prop in candidate_props:
            q = f"""
            MATCH (n:`{label}`)
            WHERE n.`{prop}` IS NOT NULL
            RETURN count(n) AS cnt,
                   count(DISTINCT n.`{prop}`) AS classes
            """
            row = session.run(q).single()
            if row and row["cnt"] > 0:
                found.append({"label": label, "prop": prop,
                              "nodes": row["cnt"], "classes": row["classes"]})

    if not found:
        result.findings.append(Finding(
            check="splits", code="nc_no_label_property",
            severity=Severity.INFO,
            message=(
                "No obvious class-label property found on any node label "
                "(looked for: " + ", ".join(candidate_props) + "). "
                "Node classification split check skipped -- configure the "
                "property explicitly if this is the target task."
            ),
        ))
        return

    for entry in found:
        label, prop = entry["label"], entry["prop"]
        class_q = f"""
        MATCH (n:`{label}`)
        WHERE n.`{prop}` IS NOT NULL
        RETURN n.`{prop}` AS cls, count(*) AS c
        ORDER BY c DESC
        """
        classes = [(r["cls"], r["c"]) for r in session.run(class_q)]
        if not classes:
            continue
        min_c = min(c for _, c in classes)
        max_c = max(c for _, c in classes)
        imbalance = max_c / min_c if min_c else float("inf")
        rare_classes = [(cls, c) for cls, c in classes if c < 10]

        sev = Severity.INFO
        notes = []
        if rare_classes:
            sev = Severity.MEDIUM
            notes.append(f"{len(rare_classes)} classes have < 10 members")
        if imbalance > 100:
            sev = Severity.MEDIUM if sev == Severity.INFO else sev
            notes.append(f"imbalance ratio {imbalance:.1f} (use stratified split)")
        note_str = "; ".join(notes) if notes else "healthy for stratified sampling"
        result.findings.append(Finding(
            check="splits", code=f"nc_label:{label}.{prop}",
            severity=sev,
            message=(
                f":{label}.{prop} has {len(classes)} classes over {entry['nodes']} "
                f"nodes. Min={min_c}, max={max_c}. {note_str}."
            ),
            count=entry["nodes"],
            details={"classes_top_20": classes[:20], "imbalance": imbalance,
                     "rare_classes": rare_classes},
        ))


def _check_entity_res_splits(session, config, result) -> None:
    # Positives come from (a) existing SAME_AS / MERGED_WITH / ALIAS_OF rels,
    # or (b) the entity_res check's candidate groups. Report both.
    known_rel_types = ["SAME_AS", "MERGED_WITH", "ALIAS_OF", "REFERS_TO_SAME"]
    positives = 0
    by_type = {}
    for rt in known_rel_types:
        row = session.run(
            f"MATCH ()-[r:`{rt}`]->() RETURN count(r) AS c"
        ).single()
        c = row["c"] if row else 0
        if c:
            by_type[rt] = c
            positives += c

    if positives == 0:
        result.findings.append(Finding(
            check="splits", code="er_no_positives",
            severity=Severity.MEDIUM,
            message=(
                "No SAME_AS / MERGED_WITH / ALIAS_OF / REFERS_TO_SAME "
                "relationships found. Entity-resolution training needs explicit "
                "positive pairs. Either (a) materialize the candidate groups "
                "from the entity_res check as SAME_AS rels and manually verify "
                "a subset, or (b) use a weak-supervision regime."
            ),
            count=0,
        ))
        return

    tr, va, te = config.split_ratios
    test_pos = int(positives * te)
    val_pos = int(positives * va)
    result.findings.append(Finding(
        check="splits", code="er_plan",
        severity=Severity.INFO if positives >= 500 else Severity.MEDIUM,
        message=(
            f"Entity-resolution positives: {positives} "
            f"({int(tr*100)}/{int(va*100)}/{int(te*100)} split = "
            f"{positives - val_pos - test_pos}/{val_pos}/{test_pos}). "
            f"{'Marginal -- ' if positives < 500 else ''}"
            f"watch transductive leakage: train pairs sharing a node with test "
            f"pairs give the model easy wins."
        ),
        count=positives,
        details={"by_relationship_type": by_type},
    ))


# ---------------------------------------------------------------------------
# Kettle-specific: Mention -> Entity ER split (the KGR supervision signal)
# ---------------------------------------------------------------------------

def _check_er_mention_entity(session, config, result) -> None:
    """The Kettle ER task: confirmed REFERS_TO is supervision, POSSIBLY_REFERS_TO
    is the promotion target. Leakage rules are strict:
        1. A Mention may appear in at most one split.
        2. An Entity may appear in at most one split (to prevent the model
           from seeing part of the cluster at train time and the rest at test).
        3. POSSIBLY_REFERS_TO edges at test time should point at Entities
           whose REFERS_TO supervision was fully held out.
    """
    from graph_diagnostics.core import lifecycle_predicate
    pred_m = lifecycle_predicate(config, var="m")
    pred_e = lifecycle_predicate(config, var="e")

    # Count confirmed vs candidate ER edges.
    q = f"""
    MATCH (m:Mention)-[r:REFERS_TO]->(e)
    WHERE {pred_m} AND {pred_e}
    RETURN count(r) AS confirmed,
           count(DISTINCT m) AS mentions,
           count(DISTINCT e) AS entities
    """
    try:
        row = session.run(q).single()
    except Exception as exc:
        result.findings.append(Finding(
            check="splits", code="er_mention_entity_query_failed",
            severity=Severity.LOW,
            message=f"REFERS_TO query failed: {exc}",
        ))
        return

    confirmed = row["confirmed"] if row else 0
    mentions = row["mentions"] if row else 0
    entities = row["entities"] if row else 0

    q2 = f"""
    MATCH (m:Mention)-[r:POSSIBLY_REFERS_TO]->(e)
    WHERE {pred_m} AND {pred_e}
    RETURN count(r) AS candidate_edges,
           count(DISTINCT m) AS candidate_mentions,
           count(DISTINCT e) AS candidate_entities
    """
    row2 = session.run(q2).single()
    candidates = row2["candidate_edges"] if row2 else 0

    if confirmed == 0:
        result.findings.append(Finding(
            check="splits", code="er_me_no_confirmed",
            severity=Severity.CRITICAL,
            message=(
                "Zero REFERS_TO edges found. The ER supervision signal is "
                "missing -- training will have nothing to learn from."
            ),
            count=0,
        ))
        return

    # Supervision-to-candidate ratio. If candidates vastly outnumber
    # confirmed, the training distribution is dominated by uncertainty.
    ratio = candidates / confirmed if confirmed else float("inf")

    result.findings.append(Finding(
        check="splits", code="er_me_supervision_summary",
        severity=Severity.INFO,
        message=(
            f"ER supervision: {confirmed:,} confirmed REFERS_TO edges "
            f"({mentions:,} mentions -> {entities:,} entities). "
            f"{candidates:,} POSSIBLY_REFERS_TO candidates pending. "
            f"Candidate/confirmed ratio: {ratio:.2f}."
        ),
        count=confirmed,
        details={
            "confirmed_edges": confirmed,
            "mentions_with_supervision": mentions,
            "entities_with_supervision": entities,
            "candidate_edges": candidates,
            "candidate_to_confirmed_ratio": ratio,
        },
    ))

    # Per-entity cluster sizes: how many mentions resolve to each entity?
    # If the distribution is dominated by singletons (one mention per entity),
    # the task is essentially candidate-classification, not cluster-linking,
    # and a simpler model may suffice.
    cluster_q = f"""
    MATCH (m:Mention)-[r:REFERS_TO]->(e)
    WHERE {pred_m} AND {pred_e}
    WITH e, count(m) AS cluster_size
    RETURN cluster_size, count(*) AS num_entities
    ORDER BY cluster_size
    """
    clusters = [(r["cluster_size"], r["num_entities"]) for r in session.run(cluster_q)]
    if clusters:
        total_entities = sum(n for _, n in clusters)
        singletons = next((n for s, n in clusters if s == 1), 0)
        max_cluster = clusters[-1][0]
        sev = Severity.INFO
        if singletons / total_entities > 0.85:
            sev = Severity.MEDIUM
        result.findings.append(Finding(
            check="splits", code="er_me_cluster_distribution",
            severity=sev,
            message=(
                f"Entity cluster sizes: {singletons}/{total_entities} "
                f"({singletons/total_entities:.1%}) entities are singletons; "
                f"max cluster = {max_cluster} mentions. "
                + ("Heavy singleton skew -- consider Mention-pair classification "
                   "rather than Entity-cluster linking." if sev == Severity.MEDIUM else
                   "Healthy cluster distribution for link-prediction framing.")
            ),
            count=total_entities,
            details={
                "cluster_size_histogram": clusters[:50],
                "singleton_share": singletons / total_entities,
                "max_cluster_size": max_cluster,
            },
        ))

    # Split leakage plan. We report what a clean split WOULD look like
    # rather than materializing it here -- materialization belongs in the
    # GNN loader.
    tr, va, te = config.split_ratios
    test_entities = int(entities * te)
    val_entities = int(entities * va)

    result.findings.append(Finding(
        check="splits", code="er_me_split_plan",
        severity=Severity.INFO,
        message=(
            f"Recommended split strategy: partition by ENTITY, not by edge. "
            f"~{entities - val_entities - test_entities} train / "
            f"{val_entities} val / {test_entities} test entities. "
            f"All REFERS_TO edges pointing at a given entity go into that "
            f"entity's split. POSSIBLY_REFERS_TO candidates targeting test "
            f"entities become the test-time promotion task. Seed: "
            f"{config.split_seed}."
        ),
        count=entities,
    ))

    # Check for entities that have BOTH confirmed and candidate supervision
    # -- these are the correct evaluation targets, but if they cluster in
    # one split the val/test metric collapses.
    both_q = f"""
    MATCH (e)
    WHERE (()-[:REFERS_TO]->(e)) AND (()-[:POSSIBLY_REFERS_TO]->(e))
      AND {pred_e}
    RETURN count(DISTINCT e) AS c
    """
    both_row = session.run(both_q).single()
    both = both_row["c"] if both_row else 0

    if both == 0:
        result.findings.append(Finding(
            check="splits", code="er_me_no_eval_targets",
            severity=Severity.HIGH,
            message=(
                "No entities have BOTH confirmed and candidate supervision. "
                "The promotion task (POSSIBLY_REFERS_TO -> REFERS_TO) cannot "
                "be evaluated: there's no entity where you can hold out "
                "confirmed edges and test whether the model re-finds them "
                "via candidates."
            ),
            count=0,
        ))
    elif both < 100:
        result.findings.append(Finding(
            check="splits", code="er_me_thin_eval_targets",
            severity=Severity.MEDIUM,
            message=(
                f"Only {both} entities have both confirmed and candidate "
                f"supervision -- the evaluation set will be thin."
            ),
            count=both,
        ))
    else:
        result.findings.append(Finding(
            check="splits", code="er_me_eval_targets",
            severity=Severity.INFO,
            message=(
                f"{both:,} entities have both confirmed and candidate "
                f"supervision. These are the natural evaluation targets: "
                f"hold out some REFERS_TO edges, see whether the model "
                f"promotes POSSIBLY_REFERS_TO candidates to close the gap."
            ),
            count=both,
        ))
