"""
Graph Builder for the Kettle Graph Reasoner synthetic data pipeline.

Given a SchemaDescriptor, generates a knowledge graph matching the measured
structural properties of real archival knowledge graphs:
    - δ = 0 (tree-like)
    - Median degree ~2
    - Mean degree ~4.6
    - Power-law degree distribution
    - Edge/node ratio ~2.28

Uses a top-down recursive growth process:
    Phase 1: Source layer seeding
    Phase 2: Claim tree growth (provenance chains)
    Phase 3: Entity population (with preferential attachment reuse)
    Phase 4: Temporal scoping
    Phase 5: Co-occurrence edges (controlled)
    Phase 6: Entity resolution planting (duplicate entities for ER tasks)

Reference: KGR Synthetic Generator Spec v0.1, Section 3.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

from schema_sampler import (
    SchemaDescriptor, EdgeTypeSpec,
    LAYER_SOURCE, LAYER_CLAIM, LAYER_ENTITY, LAYER_AUXILIARY,
    EDGE_CAT_PROVENANCE, EDGE_CAT_REFERENCE, EDGE_CAT_STRUCTURAL, EDGE_CAT_COOCCURRENCE,
)


@dataclass
class NodeData:
    """Metadata for a single node in the generated graph."""
    node_id: int
    node_type: int          # index into schema's node types
    layer: int              # LAYER_* constant
    depth: int              # depth in provenance hierarchy (0 = source)
    temporal_start: float   # normalized temporal window start [0, 1]
    temporal_end: float     # normalized temporal window end [0, 1]
    identity_vector: np.ndarray  # random 8-dim vector for pseudo-identity
    is_duplicate_of: Optional[int] = None  # if this is a planted duplicate
    duplicate_tier: Optional[int] = None   # 1=exact, 2=near, 3=structural


@dataclass
class EdgeData:
    """Metadata for a single edge in the generated graph."""
    source: int         # source node_id
    target: int         # target node_id
    edge_type: int      # index into schema's edge types
    category: int       # EDGE_CAT_* constant
    directed: bool


@dataclass 
class SyntheticGraph:
    """Complete generated graph with metadata."""
    schema: SchemaDescriptor
    nodes: Dict[int, NodeData]
    edges: List[EdgeData]
    duplicate_pairs: List[Tuple[int, int, int]]  # (original, duplicate, tier)
    seed: int
    
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def n_edges(self) -> int:
        return len(self.edges)
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph for analysis."""
        G = nx.DiGraph()
        for nid, nd in self.nodes.items():
            G.add_node(nid, node_type=nd.node_type, layer=nd.layer, depth=nd.depth)
        for ed in self.edges:
            G.add_edge(ed.source, ed.target, edge_type=ed.edge_type, category=ed.category)
            if not ed.directed:
                G.add_edge(ed.target, ed.source, edge_type=ed.edge_type, category=ed.category)
        return G
    
    def summary(self) -> str:
        """Human-readable summary."""
        layer_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for nd in self.nodes.values():
            layer_counts[nd.layer] += 1
        
        cat_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for ed in self.edges:
            cat_counts[ed.category] += 1
        
        degrees = []
        G = self.to_networkx()
        for n in G.nodes():
            degrees.append(G.degree(n))
        degrees = np.array(degrees) if degrees else np.array([0])
        
        return (
            f"SyntheticGraph (seed={self.seed})\n"
            f"  Nodes: {self.n_nodes} "
            f"(src={layer_counts[0]}, claim={layer_counts[1]}, "
            f"entity={layer_counts[2]}, aux={layer_counts[3]})\n"
            f"  Edges: {self.n_edges} "
            f"(prov={cat_counts[0]}, ref={cat_counts[1]}, "
            f"struct={cat_counts[2]}, cooc={cat_counts[3]})\n"
            f"  Edge/node ratio: {self.n_edges / max(self.n_nodes, 1):.2f}\n"
            f"  Degree: median={np.median(degrees):.1f}, "
            f"mean={np.mean(degrees):.2f}, max={np.max(degrees)}\n"
            f"  Duplicate pairs: {len(self.duplicate_pairs)} "
            f"(T1={sum(1 for _,_,t in self.duplicate_pairs if t==1)}, "
            f"T2={sum(1 for _,_,t in self.duplicate_pairs if t==2)}, "
            f"T3={sum(1 for _,_,t in self.duplicate_pairs if t==3)})"
        )


@dataclass
class BuilderConfig:
    """Configuration for graph generation."""
    target_nodes: int = 500
    max_depth: int = 5
    p_reuse: float = 0.35
    co_occurrence_rate: float = 0.05
    temporal_range: Tuple[float, float] = (0.0, 1.0)
    
    # Entity resolution planting rates
    p_dup_exact: float = 0.03     # Tier 1: exact duplicates
    p_dup_near: float = 0.03      # Tier 2: near-duplicates
    p_dup_struct: float = 0.02    # Tier 3: structural analogs
    near_dup_sigma: Tuple[float, float] = (0.05, 0.3)  # noise range for Tier 2
    
    # Distribution parameters
    branch_factor_mu: float = 1.2
    branch_factor_sigma: float = 0.8
    entities_per_claim_p: float = 0.4  # geometric distribution parameter


class GraphBuilder:
    """
    Builds synthetic knowledge graphs from schema descriptors.
    
    The generation follows a six-phase top-down growth process that
    naturally produces tree-like hierarchical graphs with power-law
    degree distributions.
    """
    
    def __init__(self, config: Optional[BuilderConfig] = None):
        self.config = config or BuilderConfig()
    
    def build(self, schema: SchemaDescriptor, seed: int) -> SyntheticGraph:
        """
        Generate a complete synthetic knowledge graph.
        
        Args:
            schema: the ontology to instantiate
            seed: deterministic random seed
            
        Returns:
            SyntheticGraph with all nodes, edges, and duplicate metadata
        """
        rng = np.random.default_rng(seed)
        cfg = self.config
        
        nodes: Dict[int, NodeData] = {}
        edges: List[EdgeData] = []
        duplicate_pairs: List[Tuple[int, int, int]] = []
        next_id = 0
        
        # Track entities by type for preferential attachment
        entities_by_type: Dict[int, List[int]] = {}
        for et in schema.get_entity_node_types():
            entities_by_type[et] = []
        
        # Track all entity IDs for reuse sampling
        all_entity_ids: List[int] = []
        entity_degrees: Dict[int, int] = {}  # node_id -> reference count
        
        # Get available edge types by category
        prov_edges = [e for e in schema.edge_specs if e.category == EDGE_CAT_PROVENANCE]
        ref_edges = [e for e in schema.edge_specs if e.category == EDGE_CAT_REFERENCE]
        struct_edges = [e for e in schema.edge_specs if e.category == EDGE_CAT_STRUCTURAL]
        cooc_edges = [e for e in schema.edge_specs if e.category == EDGE_CAT_COOCCURRENCE]
        
        # Separate provenance edges: claim->source vs claim->claim
        prov_to_source = [e for e in prov_edges if LAYER_SOURCE in e.valid_target_layers]
        prov_to_claim = [e for e in prov_edges if LAYER_CLAIM in e.valid_target_layers 
                         and LAYER_SOURCE not in e.valid_target_layers]
        # If no dedicated claim->claim edges, allow claim->source edges for all provenance
        if not prov_to_claim:
            prov_to_claim = prov_to_source
        
        def make_identity_vector(rng):
            return rng.standard_normal(8).astype(np.float32)
        
        def sample_temporal_window(rng, parent_start=None, parent_end=None):
            """Sample a temporal window, optionally constrained by parent."""
            t_lo, t_hi = cfg.temporal_range
            if parent_start is not None:
                t_lo = max(t_lo, parent_start)
            if parent_end is not None:
                t_hi = min(t_hi, parent_end)
            if t_lo >= t_hi:
                return t_lo, t_hi
            start = float(rng.uniform(t_lo, t_hi))
            end = float(rng.uniform(start, t_hi))
            return start, end
        
        def add_node(node_type, layer, depth, t_start, t_end, 
                      identity_vec=None, dup_of=None, dup_tier=None):
            nonlocal next_id
            nid = next_id
            next_id += 1
            nodes[nid] = NodeData(
                node_id=nid,
                node_type=node_type,
                layer=layer,
                depth=depth,
                temporal_start=t_start,
                temporal_end=t_end,
                identity_vector=identity_vec if identity_vec is not None else make_identity_vector(rng),
                is_duplicate_of=dup_of,
                duplicate_tier=dup_tier,
            )
            return nid
        
        def add_edge(source, target, edge_spec):
            edges.append(EdgeData(
                source=source,
                target=target,
                edge_type=edge_spec.index,
                category=edge_spec.category,
                directed=edge_spec.directed,
            ))
        
        def pick_edge_for_connection(edge_list, src_type, tgt_type, rng):
            """Find a valid edge type for connecting two specific node types."""
            valid = [e for e in edge_list 
                     if src_type in e.valid_source_types and tgt_type in e.valid_target_types]
            if not valid:
                # Fallback: any edge in the list
                valid = edge_list
            if not valid:
                return None
            return valid[int(rng.integers(0, len(valid)))]
        
        # ============================================================
        # PHASE 1: Source layer
        # ============================================================
        source_types = schema.get_source_node_types()
        n_sources = max(1, int(np.clip(
            rng.zipf(1.5) + rng.integers(2, max(3, cfg.target_nodes // 50)),
            2, cfg.target_nodes // 10
        )))
        
        source_ids = []
        for _ in range(n_sources):
            stype = source_types[int(rng.integers(0, len(source_types)))]
            t_start, t_end = sample_temporal_window(rng)
            sid = add_node(stype, LAYER_SOURCE, depth=0, 
                          t_start=t_start, t_end=t_end)
            source_ids.append(sid)
        
        # ============================================================
        # PHASE 2: Claim tree growth
        # ============================================================
        claim_types = schema.get_claim_node_types()
        
        # Queue-based tree growth from sources
        growth_queue = []  # (parent_id, current_depth)
        for sid in source_ids:
            growth_queue.append((sid, 0))
        
        claim_ids = []
        nodes_created = len(source_ids)
        target_claims = int(cfg.target_nodes * 0.35)  # ~35% of nodes are claims
        
        while growth_queue and len(claim_ids) < target_claims:
            parent_id, depth = growth_queue.pop(0)
            parent = nodes[parent_id]
            
            if depth >= cfg.max_depth:
                continue
            
            # Sample branching factor (log-normal for heavy tail)
            n_children = max(0, int(np.round(
                rng.lognormal(cfg.branch_factor_mu, cfg.branch_factor_sigma)
            )))
            # Reduce branching at deeper levels
            n_children = max(0, int(n_children * (0.7 ** depth)))
            
            # Don't exceed target
            n_children = min(n_children, target_claims - len(claim_ids))
            
            for _ in range(n_children):
                ctype = claim_types[int(rng.integers(0, len(claim_types)))]
                t_start, t_end = sample_temporal_window(
                    rng, parent.temporal_start, parent.temporal_end
                )
                
                cid = add_node(ctype, LAYER_CLAIM, depth=depth + 1,
                              t_start=t_start, t_end=t_end)
                claim_ids.append(cid)
                
                # Connect to parent with provenance edge
                if parent.layer == LAYER_SOURCE:
                    edge_spec = pick_edge_for_connection(
                        prov_to_source, ctype, parent.node_type, rng)
                else:
                    edge_spec = pick_edge_for_connection(
                        prov_to_claim, ctype, parent.node_type, rng)
                
                if edge_spec:
                    add_edge(cid, parent_id, edge_spec)
                
                # Add children to growth queue
                if depth + 1 < cfg.max_depth:
                    growth_queue.append((cid, depth + 1))
        
        # ============================================================
        # PHASE 3: Entity population
        # ============================================================
        entity_types = schema.get_entity_node_types()
        target_entities = int(cfg.target_nodes * 0.50)  # ~50% entities
        
        for cid in claim_ids:
            claim = nodes[cid]
            
            # Number of entity references per claim (geometric distribution)
            n_refs = 1 + int(rng.geometric(cfg.entities_per_claim_p))
            n_refs = min(n_refs, 4)
            
            for _ in range(n_refs):
                if (all_entity_ids and rng.random() < cfg.p_reuse 
                        and len(all_entity_ids) < target_entities):
                    # REUSE existing entity (preferential attachment)
                    weights = np.array([entity_degrees.get(eid, 1) 
                                       for eid in all_entity_ids], dtype=np.float64)
                    weights /= weights.sum()
                    eid = all_entity_ids[rng.choice(len(all_entity_ids), p=weights)]
                    
                    # Connect claim to existing entity
                    edge_spec = pick_edge_for_connection(
                        ref_edges, claim.node_type, nodes[eid].node_type, rng)
                    if edge_spec:
                        add_edge(cid, eid, edge_spec)
                        entity_degrees[eid] = entity_degrees.get(eid, 0) + 1
                
                elif len(all_entity_ids) < target_entities:
                    # CREATE new entity
                    etype = entity_types[int(rng.integers(0, len(entity_types)))]
                    t_start, t_end = sample_temporal_window(
                        rng, claim.temporal_start, claim.temporal_end
                    )
                    
                    eid = add_node(etype, LAYER_ENTITY, depth=claim.depth + 1,
                                  t_start=t_start, t_end=t_end)
                    
                    all_entity_ids.append(eid)
                    entities_by_type.setdefault(etype, []).append(eid)
                    entity_degrees[eid] = 1
                    
                    # Connect claim to new entity
                    edge_spec = pick_edge_for_connection(
                        ref_edges, claim.node_type, etype, rng)
                    if edge_spec:
                        add_edge(cid, eid, edge_spec)
        
        # ============================================================
        # PHASE 4: Temporal scoping
        # ============================================================
        auxiliary_types = schema.get_auxiliary_node_types()
        temporal_struct_edges = [e for e in struct_edges 
                                 if LAYER_AUXILIARY in e.valid_target_layers]
        
        if auxiliary_types and temporal_struct_edges:
            # Create time period nodes
            n_periods = max(3, min(20, cfg.target_nodes // 50))
            period_ids = []
            for i in range(n_periods):
                atype = auxiliary_types[int(rng.integers(0, len(auxiliary_types)))]
                t_lo = i / n_periods
                t_hi = (i + 1) / n_periods
                pid = add_node(atype, LAYER_AUXILIARY, depth=0,
                              t_start=t_lo, t_end=t_hi)
                period_ids.append((pid, t_lo, t_hi))
            
            # Connect source and claim nodes to their temporal periods
            for nid, nd in list(nodes.items()):
                if nd.layer in (LAYER_SOURCE, LAYER_CLAIM):
                    for pid, t_lo, t_hi in period_ids:
                        if nd.temporal_start < t_hi and nd.temporal_end > t_lo:
                            edge_spec = pick_edge_for_connection(
                                temporal_struct_edges, nd.node_type, 
                                nodes[pid].node_type, rng)
                            if edge_spec:
                                add_edge(nid, pid, edge_spec)
                                break  # one temporal link per node
        
        # Non-temporal structural edges (supersedes, corroborates)
        non_temporal_struct = [e for e in struct_edges 
                               if LAYER_AUXILIARY not in e.valid_target_layers]
        if non_temporal_struct:
            # Add a few structural edges between same-layer nodes
            n_struct = max(1, int(len(nodes) * 0.02))
            for _ in range(n_struct):
                es = non_temporal_struct[int(rng.integers(0, len(non_temporal_struct)))]
                src_layer = es.valid_source_layers[0]
                candidates = [nid for nid, nd in nodes.items() if nd.layer == src_layer]
                if len(candidates) >= 2:
                    a, b = rng.choice(candidates, size=2, replace=False)
                    add_edge(int(a), int(b), es)
        
        # ============================================================
        # PHASE 5: Co-occurrence edges
        # ============================================================
        if cooc_edges and all_entity_ids:
            n_cooc = max(0, int(len(all_entity_ids) * cfg.co_occurrence_rate))
            for _ in range(n_cooc):
                if len(all_entity_ids) < 2:
                    break
                a, b = rng.choice(all_entity_ids, size=2, replace=False)
                edge_spec = pick_edge_for_connection(
                    cooc_edges, nodes[int(a)].node_type, nodes[int(b)].node_type, rng)
                if edge_spec:
                    add_edge(int(a), int(b), edge_spec)
        
        # ============================================================
        # PHASE 6: Entity resolution planting
        # ============================================================
        for eid in list(all_entity_ids):
            original = nodes[eid]
            roll = float(rng.random())
            
            if roll < cfg.p_dup_exact:
                # Tier 1: Exact duplicate
                dup_id = add_node(
                    original.node_type, LAYER_ENTITY, original.depth,
                    original.temporal_start, original.temporal_end,
                    identity_vec=original.identity_vector.copy(),
                    dup_of=eid, dup_tier=1,
                )
                # Connect to a different claim than the original
                other_claims = [c for c in claim_ids 
                               if not any(e.source == c and e.target == eid for e in edges)]
                if other_claims:
                    claim_id = other_claims[int(rng.integers(0, len(other_claims)))]
                    edge_spec = pick_edge_for_connection(
                        ref_edges, nodes[claim_id].node_type, original.node_type, rng)
                    if edge_spec:
                        add_edge(claim_id, dup_id, edge_spec)
                
                duplicate_pairs.append((eid, dup_id, 1))
            
            elif roll < cfg.p_dup_exact + cfg.p_dup_near:
                # Tier 2: Near-duplicate (perturbed features)
                sigma = float(rng.uniform(cfg.near_dup_sigma[0], cfg.near_dup_sigma[1]))
                perturbed_vec = original.identity_vector + rng.normal(0, sigma, size=8).astype(np.float32)
                
                dup_id = add_node(
                    original.node_type, LAYER_ENTITY, original.depth,
                    original.temporal_start, original.temporal_end,
                    identity_vec=perturbed_vec,
                    dup_of=eid, dup_tier=2,
                )
                other_claims = [c for c in claim_ids
                               if not any(e.source == c and e.target == eid for e in edges)]
                if other_claims:
                    claim_id = other_claims[int(rng.integers(0, len(other_claims)))]
                    edge_spec = pick_edge_for_connection(
                        ref_edges, nodes[claim_id].node_type, original.node_type, rng)
                    if edge_spec:
                        add_edge(claim_id, dup_id, edge_spec)
                
                duplicate_pairs.append((eid, dup_id, 2))
            
            elif roll < cfg.p_dup_exact + cfg.p_dup_near + cfg.p_dup_struct:
                # Tier 3: Structural analog (different features, same topology)
                new_vec = make_identity_vector(rng)  # completely different features
                
                dup_id = add_node(
                    original.node_type, LAYER_ENTITY, original.depth,
                    original.temporal_start, original.temporal_end,
                    identity_vec=new_vec,
                    dup_of=eid, dup_tier=3,
                )
                # Mirror the original's connectivity pattern with different claims
                original_edges = [e for e in edges if e.target == eid and e.category == EDGE_CAT_REFERENCE]
                for orig_edge in original_edges[:3]:  # cap at 3 mirrored connections
                    # Find a similar claim (same type, different source tree)
                    similar_claims = [c for c in claim_ids 
                                     if nodes[c].node_type == nodes[orig_edge.source].node_type
                                     and c != orig_edge.source
                                     and not any(e.source == c and e.target == dup_id for e in edges)]
                    if similar_claims:
                        claim_id = similar_claims[int(rng.integers(0, len(similar_claims)))]
                        edge_spec = pick_edge_for_connection(
                            ref_edges, nodes[claim_id].node_type, original.node_type, rng)
                        if edge_spec:
                            add_edge(claim_id, dup_id, edge_spec)
                
                duplicate_pairs.append((eid, dup_id, 3))
        
        return SyntheticGraph(
            schema=schema,
            nodes=nodes,
            edges=edges,
            duplicate_pairs=duplicate_pairs,
            seed=seed,
        )


# --- Quick self-test ---
if __name__ == "__main__":
    from schema_sampler import SchemaSampler, validate_schema
    
    print("Generating test graphs...\n")
    
    sampler = SchemaSampler(master_seed=42)
    builder = GraphBuilder(BuilderConfig(target_nodes=500))
    
    for i in range(5):
        schema = sampler.sample()
        graph = builder.build(schema, seed=1000 + i)
        print(graph.summary())
        print()
    
    # Test with different sizes
    print("="*60)
    print("SIZE SCALING TEST")
    print("="*60)
    
    for target in [50, 200, 500, 1000, 2000]:
        schema = sampler.sample()
        builder_sized = GraphBuilder(BuilderConfig(target_nodes=target))
        graph = builder_sized.build(schema, seed=7777)
        print(f"  target={target:5d} -> actual={graph.n_nodes:5d} nodes, "
              f"{graph.n_edges:5d} edges, "
              f"ratio={graph.n_edges/max(graph.n_nodes,1):.2f}, "
              f"dups={len(graph.duplicate_pairs)}")
    
    # Determinism test
    print("\n" + "="*60)
    print("DETERMINISM TEST")
    print("="*60)
    schema1 = SchemaSampler(master_seed=42).sample(seed=100)
    g1 = GraphBuilder().build(schema1, seed=200)
    g2 = GraphBuilder().build(schema1, seed=200)
    assert g1.n_nodes == g2.n_nodes
    assert g1.n_edges == g2.n_edges
    assert len(g1.duplicate_pairs) == len(g2.duplicate_pairs)
    print("Deterministic reproduction confirmed.")
    
    # Tree-likeness check
    print("\n" + "="*60)
    print("TREE-LIKENESS CHECK")
    print("="*60)
    schema = SchemaSampler(master_seed=42).sample(seed=555)
    graph = GraphBuilder(BuilderConfig(target_nodes=500, co_occurrence_rate=0.0)).build(schema, seed=888)
    G = graph.to_networkx()
    G_undirected = G.to_undirected()
    
    if nx.is_connected(G_undirected):
        print(f"  Connected: yes")
        # Check cycle count (trees have exactly n-1 edges)
        n = G_undirected.number_of_nodes()
        e = G_undirected.number_of_edges()
        print(f"  Nodes: {n}, Undirected edges: {e}")
        print(f"  Excess edges (cycles): {e - (n - 1)}")
        print(f"  Tree-like: {'yes' if e - (n-1) < n * 0.1 else 'no'}")
    else:
        components = list(nx.connected_components(G_undirected))
        print(f"  Connected: no ({len(components)} components)")
        largest = max(components, key=len)
        sub = G_undirected.subgraph(largest)
        n, e = sub.number_of_nodes(), sub.number_of_edges()
        print(f"  Largest component: {n} nodes, {e} edges")
        print(f"  Excess edges: {e - (n - 1)}")
    
    print("\nAll tests passed.")
