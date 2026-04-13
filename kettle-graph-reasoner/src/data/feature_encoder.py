"""
Feature Encoder for the Kettle Graph Reasoner synthetic data pipeline.

Computes domain-agnostic feature vectors for nodes, edges, and queries.
All features use integer type indices (not string labels) to prevent
the model from learning linguistic shortcuts.

Node features: 32 dimensions
Edge features: 30 dimensions  
Query features: 9 dimensions (+ anchor pointer)

Reference: KGR Synthetic Generator Spec v0.1, Section 5.
"""

import numpy as np
from typing import Dict, List, Optional
from graph_builder import SyntheticGraph, NodeData, EdgeData
from schema_sampler import (
    SchemaDescriptor, MAX_NODE_TYPES, MAX_EDGE_TYPES,
    LAYER_SOURCE, LAYER_CLAIM, LAYER_ENTITY, LAYER_AUXILIARY,
)

# Feature dimensions
NODE_FEAT_DIM = 32
EDGE_FEAT_DIM = 30
QUERY_FEAT_DIM = 9

# Sub-dimensions
NODE_TYPE_DIM = MAX_NODE_TYPES  # 16 (one-hot, padded)
NODE_LAYER_DIM = 4              # one-hot: source, claim, entity, auxiliary
NODE_STRUCT_DIM = 5             # degree, in-degree, out-degree, clustering, depth
NODE_TEMPORAL_DIM = 3           # start, end, duration
NODE_IDENTITY_DIM = 8           # random identity vector
# Total: 16 + 4 + 5 + 3 + 8 = 36... we'll truncate type to 12 to hit 32
# Actually: let's use 12 for type one-hot to match spec exactly
NODE_TYPE_DIM_ACTUAL = 12

EDGE_TYPE_DIM = 25              # one-hot for edge type (padded)
EDGE_CAT_DIM = 4               # provenance, reference, structural, co-occurrence
EDGE_DIR_DIM = 1               # directed flag
# 25 + 4 + 1 = 30 ✓

QUERY_TASK_DIM = 5             # one-hot: 5 task types
QUERY_TEMPORAL_DIM = 2         # start, end
QUERY_HOPS_DIM = 1             # max hops (normalized)
QUERY_PAD_DIM = 1              # padding to 9


def encode_nodes(graph: SyntheticGraph) -> np.ndarray:
    """
    Encode all nodes as a (N x NODE_FEAT_DIM) feature matrix.
    
    Feature layout:
        [0:12]   - node type one-hot (padded to 12)
        [12:16]  - layer one-hot (4 dims)
        [16:21]  - structural features (degree, in-deg, out-deg, clustering, depth)
        [21:24]  - temporal features (start, end, duration)
        [24:32]  - identity vector (8 dims)
    """
    n = graph.n_nodes
    features = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)
    
    # Precompute degree information from edges
    in_degree = np.zeros(n, dtype=np.float32)
    out_degree = np.zeros(n, dtype=np.float32)
    neighbor_sets: Dict[int, set] = {i: set() for i in range(n)}
    
    for ed in graph.edges:
        if ed.source < n and ed.target < n:
            out_degree[ed.source] += 1
            in_degree[ed.target] += 1
            neighbor_sets[ed.source].add(ed.target)
            neighbor_sets[ed.target].add(ed.source)
            if not ed.directed:
                out_degree[ed.target] += 1
                in_degree[ed.source] += 1
    
    total_degree = in_degree + out_degree
    
    # Compute local clustering coefficient per node
    clustering = np.zeros(n, dtype=np.float32)
    for nid in range(n):
        neighbors = list(neighbor_sets[nid])
        k = len(neighbors)
        if k < 2:
            continue
        # Count edges among neighbors
        links = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in neighbor_sets[neighbors[i]]:
                    links += 1
        clustering[nid] = 2.0 * links / (k * (k - 1))
    
    # Build feature matrix
    # Use sorted node IDs to ensure consistent ordering
    id_to_row = {}
    for i, nid in enumerate(sorted(graph.nodes.keys())):
        id_to_row[nid] = i
    
    for nid in sorted(graph.nodes.keys()):
        nd = graph.nodes[nid]
        row = id_to_row[nid]
        
        # Node type one-hot (12 dims)
        if nd.node_type < NODE_TYPE_DIM_ACTUAL:
            features[row, nd.node_type] = 1.0
        
        # Layer one-hot (4 dims)
        features[row, NODE_TYPE_DIM_ACTUAL + nd.layer] = 1.0
        
        # Structural features (5 dims)
        struct_offset = NODE_TYPE_DIM_ACTUAL + NODE_LAYER_DIM
        features[row, struct_offset + 0] = np.log1p(total_degree[row])   # log degree
        features[row, struct_offset + 1] = np.log1p(in_degree[row])      # log in-degree
        features[row, struct_offset + 2] = np.log1p(out_degree[row])     # log out-degree
        features[row, struct_offset + 3] = clustering[row]                # clustering coeff
        features[row, struct_offset + 4] = nd.depth / 10.0               # normalized depth
        
        # Temporal features (3 dims)
        temp_offset = struct_offset + NODE_STRUCT_DIM
        features[row, temp_offset + 0] = nd.temporal_start
        features[row, temp_offset + 1] = nd.temporal_end
        features[row, temp_offset + 2] = nd.temporal_end - nd.temporal_start  # duration
        
        # Identity vector (8 dims)
        id_offset = temp_offset + NODE_TEMPORAL_DIM
        features[row, id_offset:id_offset + 8] = nd.identity_vector
    
    return features, id_to_row


def encode_edges(graph: SyntheticGraph, id_to_row: Dict[int, int]) -> tuple:
    """
    Encode all edges as:
        - edge_index: (2, E) array of (source_row, target_row) pairs
        - edge_attr: (E, EDGE_FEAT_DIM) feature matrix
    
    Feature layout:
        [0:25]   - edge type one-hot (padded to 25)
        [25:29]  - category one-hot (4 dims)
        [29]     - direction flag
    """
    E = graph.n_edges
    edge_index = np.zeros((2, E), dtype=np.int64)
    edge_attr = np.zeros((E, EDGE_FEAT_DIM), dtype=np.float32)
    
    for i, ed in enumerate(graph.edges):
        src_row = id_to_row.get(ed.source, 0)
        tgt_row = id_to_row.get(ed.target, 0)
        
        edge_index[0, i] = src_row
        edge_index[1, i] = tgt_row
        
        # Edge type one-hot (25 dims)
        if ed.edge_type < EDGE_TYPE_DIM:
            edge_attr[i, ed.edge_type] = 1.0
        
        # Category one-hot (4 dims)
        edge_attr[i, EDGE_TYPE_DIM + ed.category] = 1.0
        
        # Direction flag
        edge_attr[i, EDGE_TYPE_DIM + 4] = 1.0 if ed.directed else 0.0
    
    return edge_index, edge_attr


def encode_query(task_type: int, anchor_row: int,
                 temporal_window: Optional[tuple] = None,
                 max_hops: int = 4) -> np.ndarray:
    """
    Encode a query as a fixed-size feature vector.
    
    Args:
        task_type: 0-4 corresponding to the five task types
        anchor_row: row index of the anchor node in the feature matrix
        temporal_window: (start, end) or None
        max_hops: maximum hop distance
    
    Returns:
        (QUERY_FEAT_DIM,) feature vector
    
    Feature layout:
        [0:5]  - task type one-hot
        [5:7]  - temporal window (start, end)
        [7]    - max hops (normalized by 10)
        [8]    - padding
    """
    q = np.zeros(QUERY_FEAT_DIM, dtype=np.float32)
    
    # Task type one-hot
    if 0 <= task_type < QUERY_TASK_DIM:
        q[task_type] = 1.0
    
    # Temporal window
    if temporal_window is not None:
        q[5] = temporal_window[0]
        q[6] = temporal_window[1]
    
    # Max hops (normalized)
    q[7] = max_hops / 10.0
    
    return q


# --- Self-test ---
if __name__ == "__main__":
    from schema_sampler import SchemaSampler
    from graph_builder import GraphBuilder, BuilderConfig
    
    sampler = SchemaSampler(master_seed=42)
    schema = sampler.sample(seed=100)
    builder = GraphBuilder(BuilderConfig(target_nodes=200))
    graph = builder.build(schema, seed=200)
    
    print(f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges\n")
    
    # Encode nodes
    node_features, id_to_row = encode_nodes(graph)
    print(f"Node features: shape={node_features.shape}, dtype={node_features.dtype}")
    print(f"  Expected: ({graph.n_nodes}, {NODE_FEAT_DIM})")
    assert node_features.shape == (graph.n_nodes, NODE_FEAT_DIM)
    
    # Check no NaN or Inf
    assert np.all(np.isfinite(node_features)), "Node features contain NaN/Inf!"
    
    # Encode edges
    edge_index, edge_attr = encode_edges(graph, id_to_row)
    print(f"\nEdge index: shape={edge_index.shape}")
    print(f"Edge attr:  shape={edge_attr.shape}, dtype={edge_attr.dtype}")
    print(f"  Expected: (2, {graph.n_edges}) and ({graph.n_edges}, {EDGE_FEAT_DIM})")
    assert edge_index.shape == (2, graph.n_edges)
    assert edge_attr.shape == (graph.n_edges, EDGE_FEAT_DIM)
    assert np.all(np.isfinite(edge_attr))
    
    # Encode query
    query = encode_query(task_type=0, anchor_row=5, 
                         temporal_window=(0.2, 0.8), max_hops=3)
    print(f"\nQuery features: shape={query.shape}")
    assert query.shape == (QUERY_FEAT_DIM,)
    
    # Check feature statistics
    print(f"\nNode feature stats:")
    print(f"  Mean: {node_features.mean():.4f}")
    print(f"  Std:  {node_features.std():.4f}")
    print(f"  Min:  {node_features.min():.4f}")
    print(f"  Max:  {node_features.max():.4f}")
    
    print(f"\nEdge feature stats:")
    print(f"  Mean: {edge_attr.mean():.4f}")
    print(f"  Std:  {edge_attr.std():.4f}")
    
    print("\nAll feature encoding tests passed.")
