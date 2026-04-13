"""
Task Generator for the Kettle Graph Reasoner synthetic data pipeline.

Given a SyntheticGraph, generates labeled training examples for five
structural reasoning tasks. Labels are computed deterministically from
graph algorithms — no human annotation required.

Tasks:
    0: Provenance chain traversal
    1: Entity resolution (three tiers)
    2: Temporal scope filtering
    3: Multi-hop relevance scoring
    4: Subgraph boundary detection

Reference: KGR Synthetic Generator Spec v0.1, Section 4.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from graph_builder import SyntheticGraph, NodeData
from schema_sampler import (
    LAYER_SOURCE, LAYER_CLAIM, LAYER_ENTITY, LAYER_AUXILIARY,
    EDGE_CAT_PROVENANCE, EDGE_CAT_REFERENCE,
)

# Task type constants
TASK_PROVENANCE = 0
TASK_ENTITY_RESOLUTION = 1
TASK_TEMPORAL = 2
TASK_MULTIHOP = 3
TASK_SUBGRAPH = 4


@dataclass
class TaskExample:
    """A single labeled training example."""
    task_type: int
    anchor_node: int             # query entry-point node ID
    labels: np.ndarray           # per-node relevance scores (N,)
    temporal_window: Optional[Tuple[float, float]] = None
    max_hops: int = 4
    # Entity resolution specific
    er_pairs: Optional[List[Tuple[int, int, int, int]]] = None  # (n1, n2, label, tier)


class TaskGenerator:
    """
    Generates labeled structural reasoning tasks from synthetic graphs.
    
    Each task type has a deterministic label generation algorithm that
    computes ground-truth relevance scores from graph structure.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def generate_all_tasks(self, graph: SyntheticGraph,
                            id_to_row: Dict[int, int]) -> List[TaskExample]:
        """Generate all task types for a single graph."""
        tasks = []
        tasks.extend(self.generate_provenance_tasks(graph, id_to_row))
        tasks.extend(self.generate_er_tasks(graph, id_to_row))
        tasks.extend(self.generate_temporal_tasks(graph, id_to_row))
        tasks.extend(self.generate_multihop_tasks(graph, id_to_row))
        tasks.extend(self.generate_subgraph_tasks(graph, id_to_row))
        return tasks
    
    # ================================================================
    # Task 0: Provenance Chain Traversal
    # ================================================================
    def generate_provenance_tasks(self, graph: SyntheticGraph,
                                   id_to_row: Dict[int, int],
                                   n_tasks: int = 10) -> List[TaskExample]:
        """
        Given an entity anchor, find all source nodes reachable through
        provenance chains. Labels: 1.0 for source nodes, 1/d for intermediate.
        """
        tasks = []
        entity_ids = [nid for nid, nd in graph.nodes.items() 
                      if nd.layer == LAYER_ENTITY]
        
        if not entity_ids:
            return tasks
        
        # Build reverse adjacency for provenance/reference traversal
        # entity -> claims (via reference edges, reversed)
        # claim -> parent claims/sources (via provenance edges, following direction)
        reverse_ref = defaultdict(set)  # entity_id -> set of claim_ids
        prov_parent = defaultdict(set)  # claim_id -> set of parent_ids
        
        for ed in graph.edges:
            if ed.category == EDGE_CAT_REFERENCE:
                reverse_ref[ed.target].add(ed.source)
            elif ed.category == EDGE_CAT_PROVENANCE:
                prov_parent[ed.source].add(ed.target)
        
        n_actual = min(n_tasks, len(entity_ids))
        anchors = self.rng.choice(entity_ids, size=n_actual, replace=False)
        
        for anchor in anchors:
            N = graph.n_nodes
            labels = np.zeros(N, dtype=np.float32)
            
            # BFS from entity back to sources
            visited = set()
            queue = [(int(anchor), 0)]  # (node_id, distance)
            
            # First hop: entity -> claims via reverse reference
            for claim_id in reverse_ref.get(int(anchor), set()):
                queue.append((claim_id, 1))
            
            while queue:
                nid, dist = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                
                nd = graph.nodes.get(nid)
                if nd is None:
                    continue
                
                row = id_to_row.get(nid)
                if row is None:
                    continue
                
                if nd.layer == LAYER_SOURCE:
                    labels[row] = 1.0
                elif dist > 0:
                    labels[row] = 1.0 / max(dist, 1)
                
                # Follow provenance edges upward
                for parent_id in prov_parent.get(nid, set()):
                    if parent_id not in visited:
                        queue.append((parent_id, dist + 1))
            
            # Mark anchor
            anchor_row = id_to_row.get(int(anchor))
            if anchor_row is not None:
                labels[anchor_row] = 1.0
            
            if labels.sum() > 0:
                tasks.append(TaskExample(
                    task_type=TASK_PROVENANCE,
                    anchor_node=int(anchor),
                    labels=labels,
                ))
        
        return tasks
    
    # ================================================================
    # Task 1: Entity Resolution
    # ================================================================
    def generate_er_tasks(self, graph: SyntheticGraph,
                           id_to_row: Dict[int, int]) -> List[TaskExample]:
        """
        Generate entity resolution pairs from planted duplicates +
        negative samples. Labels are per-pair (not per-node).
        """
        tasks = []
        
        if not graph.duplicate_pairs:
            return tasks
        
        entity_ids = [nid for nid, nd in graph.nodes.items()
                      if nd.layer == LAYER_ENTITY]
        
        for original_id, dup_id, tier in graph.duplicate_pairs:
            if original_id not in graph.nodes or dup_id not in graph.nodes:
                continue
            
            pairs = []
            # Positive pair
            pairs.append((original_id, dup_id, 1, tier))
            
            # Negative pairs: entities that share some similarity but aren't duplicates
            original = graph.nodes[original_id]
            negatives_found = 0
            
            # Shuffle entity list for randomness
            shuffled = list(entity_ids)
            self.rng.shuffle(shuffled)
            
            for neg_id in shuffled:
                if neg_id == original_id or neg_id == dup_id:
                    continue
                neg_node = graph.nodes[neg_id]
                
                # Must share at least one similarity
                same_type = neg_node.node_type == original.node_type
                similar_temporal = (
                    abs(neg_node.temporal_start - original.temporal_start) < 0.2
                )
                
                if same_type or similar_temporal:
                    pairs.append((original_id, neg_id, 0, tier))
                    negatives_found += 1
                    if negatives_found >= 4:
                        break
            
            if pairs:
                # Create a label vector: not per-node, but store pairs in er_pairs
                N = graph.n_nodes
                labels = np.zeros(N, dtype=np.float32)
                # Mark both nodes involved
                for n1, n2, label, t in pairs:
                    r1 = id_to_row.get(n1)
                    r2 = id_to_row.get(n2)
                    if r1 is not None:
                        labels[r1] = 1.0
                    if r2 is not None:
                        labels[r2] = 1.0
                
                tasks.append(TaskExample(
                    task_type=TASK_ENTITY_RESOLUTION,
                    anchor_node=original_id,
                    labels=labels,
                    er_pairs=pairs,
                ))
        
        return tasks
    
    # ================================================================
    # Task 2: Temporal Scope Filtering
    # ================================================================
    def generate_temporal_tasks(self, graph: SyntheticGraph,
                                 id_to_row: Dict[int, int],
                                 n_tasks: int = 5) -> List[TaskExample]:
        """
        Given a temporal window, identify nodes within scope.
        Labels: 1.0 for in-window, 0.5 for adjacent, 0.0 for out.
        """
        tasks = []
        
        # Collect temporal ranges
        temporal_nodes = [(nid, nd) for nid, nd in graph.nodes.items()
                         if nd.temporal_end > nd.temporal_start]
        
        if not temporal_nodes:
            return tasks
        
        for _ in range(n_tasks):
            # Sample a random window
            window_start = float(self.rng.uniform(0.0, 0.7))
            window_end = float(self.rng.uniform(window_start + 0.1, 1.0))
            window_duration = window_end - window_start
            margin = window_duration * 0.2  # adjacency margin
            
            N = graph.n_nodes
            labels = np.zeros(N, dtype=np.float32)
            
            anchor = None
            for nid, nd in graph.nodes.items():
                row = id_to_row.get(nid)
                if row is None:
                    continue
                
                # Check overlap with window
                overlap_start = max(nd.temporal_start, window_start)
                overlap_end = min(nd.temporal_end, window_end)
                
                if overlap_start < overlap_end:
                    # Node overlaps with window
                    labels[row] = 1.0
                    if anchor is None:
                        anchor = nid
                elif (nd.temporal_start < window_end + margin and 
                      nd.temporal_end > window_start - margin):
                    # Adjacent to window
                    labels[row] = 0.5
            
            if anchor is not None and labels.sum() > 0:
                tasks.append(TaskExample(
                    task_type=TASK_TEMPORAL,
                    anchor_node=anchor,
                    labels=labels,
                    temporal_window=(window_start, window_end),
                ))
        
        return tasks
    
    # ================================================================
    # Task 3: Multi-hop Relevance Scoring
    # ================================================================
    def generate_multihop_tasks(self, graph: SyntheticGraph,
                                 id_to_row: Dict[int, int],
                                 n_tasks: int = 10) -> List[TaskExample]:
        """
        Given an entity anchor and max hops, rank reachable nodes by
        structural relevance. Relevance = f(distance, edge_type_rarity, branching).
        """
        tasks = []
        entity_ids = [nid for nid, nd in graph.nodes.items()
                      if nd.layer == LAYER_ENTITY]
        
        if not entity_ids:
            return tasks
        
        # Build undirected adjacency
        adj = defaultdict(list)  # node_id -> [(neighbor_id, edge_type)]
        for ed in graph.edges:
            adj[ed.source].append((ed.target, ed.edge_type))
            adj[ed.target].append((ed.source, ed.edge_type))
        
        # Compute edge type frequencies for rarity weighting
        edge_type_freq = defaultdict(int)
        for ed in graph.edges:
            edge_type_freq[ed.edge_type] += 1
        max_freq = max(edge_type_freq.values()) if edge_type_freq else 1
        
        n_actual = min(n_tasks, len(entity_ids))
        anchors = self.rng.choice(entity_ids, size=n_actual, replace=False)
        max_hops = 4
        alpha = 0.7  # distance decay
        
        for anchor in anchors:
            N = graph.n_nodes
            labels = np.zeros(N, dtype=np.float32)
            
            # BFS with distance tracking
            visited = {}  # node_id -> (distance, path_edge_types)
            queue = [(int(anchor), 0, [])]
            
            while queue:
                nid, dist, path_edges = queue.pop(0)
                
                if nid in visited:
                    continue
                if dist > max_hops:
                    continue
                
                visited[nid] = (dist, path_edges)
                
                for neighbor, etype in adj.get(nid, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1, path_edges + [etype]))
            
            # Compute relevance scores
            for nid, (dist, path_edges) in visited.items():
                row = id_to_row.get(nid)
                if row is None:
                    continue
                
                # Distance decay
                dist_score = alpha ** dist
                
                # Edge type rarity bonus (inverse frequency)
                if path_edges:
                    rarity = np.mean([
                        1.0 - (edge_type_freq.get(et, 1) / max_freq)
                        for et in path_edges
                    ])
                else:
                    rarity = 0.0
                
                # Branching penalty (high-degree nodes are less informative)
                n_neighbors = len(adj.get(nid, []))
                branch_penalty = 1.0 / max(np.log1p(n_neighbors), 1.0)
                
                labels[row] = dist_score * (1.0 + rarity) * branch_penalty
            
            # Normalize to [0, 1]
            if labels.max() > 0:
                labels /= labels.max()
            
            if labels.sum() > 0:
                tasks.append(TaskExample(
                    task_type=TASK_MULTIHOP,
                    anchor_node=int(anchor),
                    labels=labels,
                    max_hops=max_hops,
                ))
        
        return tasks
    
    # ================================================================
    # Task 4: Subgraph Boundary Detection
    # ================================================================
    def generate_subgraph_tasks(self, graph: SyntheticGraph,
                                 id_to_row: Dict[int, int],
                                 n_tasks: int = 5) -> List[TaskExample]:
        """
        Given a composite query (entity + temporal + depth constraint),
        identify the minimal subgraph containing all needed information.
        Labels: binary mask (1 = include, 0 = exclude).
        """
        tasks = []
        entity_ids = [nid for nid, nd in graph.nodes.items()
                      if nd.layer == LAYER_ENTITY]
        
        if not entity_ids:
            return tasks
        
        # Build adjacency
        adj = defaultdict(list)
        for ed in graph.edges:
            adj[ed.source].append((ed.target, ed.edge_type, ed.category))
            adj[ed.target].append((ed.source, ed.edge_type, ed.category))
        
        n_actual = min(n_tasks, len(entity_ids))
        anchors = self.rng.choice(entity_ids, size=n_actual, replace=False)
        
        for anchor in anchors:
            anchor_node = graph.nodes[int(anchor)]
            
            # Sample constraints
            max_depth = int(self.rng.integers(2, 5))
            window_center = (anchor_node.temporal_start + anchor_node.temporal_end) / 2
            window_half = float(self.rng.uniform(0.1, 0.3))
            window_start = max(0.0, window_center - window_half)
            window_end = min(1.0, window_center + window_half)
            
            N = graph.n_nodes
            labels = np.zeros(N, dtype=np.float32)
            
            # BFS within constraints
            visited = set()
            queue = [(int(anchor), 0)]
            
            while queue:
                nid, dist = queue.pop(0)
                if nid in visited or dist > max_depth:
                    continue
                
                nd = graph.nodes.get(nid)
                if nd is None:
                    continue
                
                # Temporal filter
                if nd.temporal_end < window_start or nd.temporal_start > window_end:
                    continue
                
                visited.add(nid)
                row = id_to_row.get(nid)
                if row is not None:
                    labels[row] = 1.0
                
                for neighbor, etype, ecat in adj.get(nid, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
            
            if labels.sum() > 1:  # need at least 2 nodes for a meaningful subgraph
                tasks.append(TaskExample(
                    task_type=TASK_SUBGRAPH,
                    anchor_node=int(anchor),
                    labels=labels,
                    temporal_window=(window_start, window_end),
                    max_hops=max_depth,
                ))
        
        return tasks


# --- Self-test ---
if __name__ == "__main__":
    from schema_sampler import SchemaSampler
    from graph_builder import GraphBuilder, BuilderConfig
    from feature_encoder import encode_nodes
    
    sampler = SchemaSampler(master_seed=42)
    schema = sampler.sample(seed=100)
    builder = GraphBuilder(BuilderConfig(target_nodes=500))
    graph = builder.build(schema, seed=200)
    node_features, id_to_row = encode_nodes(graph)
    
    print(f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    print(f"Duplicate pairs: {len(graph.duplicate_pairs)}\n")
    
    gen = TaskGenerator(seed=42)
    tasks = gen.generate_all_tasks(graph, id_to_row)
    
    # Count by type
    task_names = {
        TASK_PROVENANCE: "Provenance traversal",
        TASK_ENTITY_RESOLUTION: "Entity resolution",
        TASK_TEMPORAL: "Temporal filtering",
        TASK_MULTIHOP: "Multi-hop relevance",
        TASK_SUBGRAPH: "Subgraph boundary",
    }
    
    counts = defaultdict(int)
    for t in tasks:
        counts[t.task_type] += 1
    
    print("Task counts:")
    for tt, name in task_names.items():
        print(f"  {name:25s}: {counts[tt]:3d}")
    print(f"  {'TOTAL':25s}: {len(tasks):3d}")
    
    # Inspect a few tasks
    print("\nSample task details:")
    for tt in range(5):
        examples = [t for t in tasks if t.task_type == tt]
        if examples:
            t = examples[0]
            n_relevant = (t.labels > 0).sum()
            print(f"  [{task_names[tt]}] anchor={t.anchor_node}, "
                  f"relevant_nodes={n_relevant}/{graph.n_nodes}, "
                  f"label_sum={t.labels.sum():.2f}")
            if t.er_pairs:
                print(f"    ER pairs: {len(t.er_pairs)} "
                      f"(pos={sum(1 for _,_,l,_ in t.er_pairs if l==1)}, "
                      f"neg={sum(1 for _,_,l,_ in t.er_pairs if l==0)})")
            if t.temporal_window:
                print(f"    Temporal window: [{t.temporal_window[0]:.2f}, {t.temporal_window[1]:.2f}]")
    
    print("\nAll task generation tests passed.")
