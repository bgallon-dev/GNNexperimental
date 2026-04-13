"""
Schema Sampler for the Kettle Graph Reasoner synthetic data pipeline.

Generates random ontologies (node types, edge types, hierarchy rules, constraints)
parameterized by structural invariants derived from real archival knowledge graphs.
Each schema is represented as a domain-agnostic descriptor using integer indices,
not string labels, to prevent the model from learning linguistic shortcuts.

Reference: KGR Synthetic Generator Spec v0.1, Section 2.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# Hierarchical layer constants
LAYER_SOURCE = 0
LAYER_CLAIM = 1
LAYER_ENTITY = 2
LAYER_AUXILIARY = 3

LAYER_NAMES = {
    LAYER_SOURCE: "source",
    LAYER_CLAIM: "claim",
    LAYER_ENTITY: "entity",
    LAYER_AUXILIARY: "auxiliary",
}

# Edge category constants
EDGE_CAT_PROVENANCE = 0    # source <-> claim derivation chains
EDGE_CAT_REFERENCE = 1     # claim -> entity references
EDGE_CAT_STRUCTURAL = 2    # temporal scoping, supersedes, corroborates
EDGE_CAT_COOCCURRENCE = 3  # entity <-> entity co-mention

EDGE_CAT_NAMES = {
    EDGE_CAT_PROVENANCE: "provenance",
    EDGE_CAT_REFERENCE: "reference",
    EDGE_CAT_STRUCTURAL: "structural",
    EDGE_CAT_COOCCURRENCE: "co-occurrence",
}

# Maximum dimensions for padding (model receives fixed-size descriptors)
MAX_NODE_TYPES = 16
MAX_EDGE_TYPES = 30


@dataclass
class EdgeTypeSpec:
    """Specification for a single edge type within a schema."""
    index: int
    category: int                    # EDGE_CAT_* constant
    valid_source_layers: List[int]   # which node layers can be the source
    valid_target_layers: List[int]   # which node layers can be the target
    valid_source_types: List[int]    # specific node type indices (within layer)
    valid_target_types: List[int]    # specific node type indices (within layer)
    directed: bool                   # True for provenance/reference, False for co-occurrence


@dataclass
class SchemaDescriptor:
    """
    Complete schema for a synthetic knowledge graph.
    
    All type information uses integer indices. The model receives this
    descriptor alongside the graph to enable schema-portable reasoning.
    """
    # Type counts
    n_source_types: int
    n_claim_types: int
    n_entity_types: int
    n_auxiliary_types: int
    n_node_types: int  # sum of above
    n_edge_types: int
    
    # Node type -> layer mapping (length = n_node_types)
    node_layer_assignment: List[int]
    
    # Edge specifications (length = n_edge_types)
    edge_specs: List[EdgeTypeSpec]
    
    # Per-category edge type counts
    n_provenance_edges: int
    n_reference_edges: int
    n_structural_edges: int
    n_cooccurrence_edges: int
    
    # Generation metadata
    seed: int
    
    def get_node_types_for_layer(self, layer: int) -> List[int]:
        """Return all node type indices assigned to a given layer."""
        return [i for i, l in enumerate(self.node_layer_assignment) if l == layer]
    
    def get_edge_types_for_category(self, category: int) -> List[int]:
        """Return all edge type indices in a given category."""
        return [e.index for e in self.edge_specs if e.category == category]
    
    def get_source_node_types(self) -> List[int]:
        return self.get_node_types_for_layer(LAYER_SOURCE)
    
    def get_claim_node_types(self) -> List[int]:
        return self.get_node_types_for_layer(LAYER_CLAIM)
    
    def get_entity_node_types(self) -> List[int]:
        return self.get_node_types_for_layer(LAYER_ENTITY)
    
    def get_auxiliary_node_types(self) -> List[int]:
        return self.get_node_types_for_layer(LAYER_AUXILIARY)
    
    def to_tensor_dict(self) -> Dict[str, "np.ndarray"]:
        """
        Convert schema to fixed-size numpy arrays for model input.
        Padded to MAX_NODE_TYPES / MAX_EDGE_TYPES.
        """
        # Node layer assignment, padded
        nla = np.zeros(MAX_NODE_TYPES, dtype=np.int64)
        nla[:self.n_node_types] = self.node_layer_assignment
        
        # Edge category assignment, padded
        eca = np.zeros(MAX_EDGE_TYPES, dtype=np.int64)
        for e in self.edge_specs:
            eca[e.index] = e.category
        
        # Edge direction flags, padded
        edir = np.zeros(MAX_EDGE_TYPES, dtype=np.float32)
        for e in self.edge_specs:
            edir[e.index] = 1.0 if e.directed else 0.0
        
        # Edge source/target layer constraints as matrices
        # Shape: (MAX_EDGE_TYPES, 4) — one-hot over layers
        esrc_layers = np.zeros((MAX_EDGE_TYPES, 4), dtype=np.float32)
        etgt_layers = np.zeros((MAX_EDGE_TYPES, 4), dtype=np.float32)
        for e in self.edge_specs:
            for l in e.valid_source_layers:
                esrc_layers[e.index, l] = 1.0
            for l in e.valid_target_layers:
                etgt_layers[e.index, l] = 1.0
        
        return {
            "n_node_types": np.array(self.n_node_types, dtype=np.int64),
            "n_edge_types": np.array(self.n_edge_types, dtype=np.int64),
            "node_layer_assignment": nla,
            "edge_category": eca,
            "edge_directed": edir,
            "edge_source_layers": esrc_layers,
            "edge_target_layers": etgt_layers,
        }
    
    def summary(self) -> str:
        """Human-readable summary for debugging and inspection."""
        lines = [
            f"SchemaDescriptor (seed={self.seed})",
            f"  Node types: {self.n_node_types} "
            f"({self.n_source_types}S + {self.n_claim_types}C + "
            f"{self.n_entity_types}E + {self.n_auxiliary_types}A)",
            f"  Edge types: {self.n_edge_types} "
            f"({self.n_provenance_edges}prov + {self.n_reference_edges}ref + "
            f"{self.n_structural_edges}struct + {self.n_cooccurrence_edges}cooc)",
            f"  Layer assignment: {self.node_layer_assignment}",
            f"  Edge details:",
        ]
        for e in self.edge_specs:
            cat = EDGE_CAT_NAMES[e.category]
            direction = "directed" if e.directed else "undirected"
            src_layers = [LAYER_NAMES[l] for l in e.valid_source_layers]
            tgt_layers = [LAYER_NAMES[l] for l in e.valid_target_layers]
            lines.append(
                f"    [{e.index:2d}] {cat:14s} {direction:10s} "
                f"{src_layers} -> {tgt_layers}"
            )
        return "\n".join(lines)


class SchemaSampler:
    """
    Generates random schema descriptors for synthetic knowledge graphs.
    
    Structural invariants (always enforced):
        - At least 1 source type (layer 0)
        - At least 1 claim type (layer 1)  
        - At least 2 entity types (layer 2)
        - Directional hierarchy: SOURCE -> CLAIM -> ENTITY
        - At least 1 provenance edge type
        - At least 1 reference edge type
        - At least 1 structural (temporal) edge type
    
    Variable parameters (sampled per schema):
        - Number of types per layer
        - Number of edge types per category
        - Specific source/target constraints per edge type
    """
    
    # Parameter ranges from spec Section 2.3
    SOURCE_TYPES_RANGE = (1, 3)
    CLAIM_TYPES_RANGE = (1, 4)
    ENTITY_TYPES_RANGE = (2, 6)
    AUXILIARY_TYPES_RANGE = (0, 3)
    
    PROVENANCE_EDGES_RANGE = (1, 3)
    REFERENCE_EDGES_RANGE = (2, 6)
    STRUCTURAL_EDGES_RANGE = (1, 4)
    COOCCURRENCE_EDGES_RANGE = (0, 3)
    
    def __init__(self, master_seed: int = 42):
        """
        Args:
            master_seed: seed for the sampler's own RNG. Individual schemas
                        get derived seeds for reproducibility.
        """
        self.master_seed = master_seed
        self.rng = np.random.default_rng(master_seed)
        self._schema_counter = 0
    
    def sample(self, seed: Optional[int] = None) -> SchemaDescriptor:
        """
        Generate a single random schema.
        
        Args:
            seed: explicit seed for this schema. If None, derived from
                  the master RNG sequence.
        
        Returns:
            SchemaDescriptor with all type and constraint information.
        """
        if seed is None:
            seed = int(self.rng.integers(0, 2**31))
        
        rng = np.random.default_rng(seed)
        self._schema_counter += 1
        
        # --- Sample node type counts ---
        n_source = int(rng.integers(self.SOURCE_TYPES_RANGE[0], 
                                     self.SOURCE_TYPES_RANGE[1] + 1))
        n_claim = int(rng.integers(self.CLAIM_TYPES_RANGE[0], 
                                    self.CLAIM_TYPES_RANGE[1] + 1))
        n_entity = int(rng.integers(self.ENTITY_TYPES_RANGE[0], 
                                     self.ENTITY_TYPES_RANGE[1] + 1))
        n_auxiliary = int(rng.integers(self.AUXILIARY_TYPES_RANGE[0], 
                                       self.AUXILIARY_TYPES_RANGE[1] + 1))
        
        n_node_types = n_source + n_claim + n_entity + n_auxiliary
        
        # Enforce MAX_NODE_TYPES ceiling
        if n_node_types > MAX_NODE_TYPES:
            # Trim entity types first (largest range), then auxiliary
            excess = n_node_types - MAX_NODE_TYPES
            trim_aux = min(excess, n_auxiliary)
            n_auxiliary -= trim_aux
            excess -= trim_aux
            if excess > 0:
                n_entity = max(2, n_entity - excess)
            n_node_types = n_source + n_claim + n_entity + n_auxiliary
        
        # Build layer assignment array
        # Indices: [0..n_source-1] = source, [n_source..n_source+n_claim-1] = claim, etc.
        node_layer_assignment = (
            [LAYER_SOURCE] * n_source +
            [LAYER_CLAIM] * n_claim +
            [LAYER_ENTITY] * n_entity +
            [LAYER_AUXILIARY] * n_auxiliary
        )
        
        # --- Sample edge type counts ---
        n_prov = int(rng.integers(self.PROVENANCE_EDGES_RANGE[0], 
                                   self.PROVENANCE_EDGES_RANGE[1] + 1))
        n_ref = int(rng.integers(self.REFERENCE_EDGES_RANGE[0], 
                                  self.REFERENCE_EDGES_RANGE[1] + 1))
        n_struct = int(rng.integers(self.STRUCTURAL_EDGES_RANGE[0], 
                                     self.STRUCTURAL_EDGES_RANGE[1] + 1))
        n_cooc = int(rng.integers(self.COOCCURRENCE_EDGES_RANGE[0], 
                                   self.COOCCURRENCE_EDGES_RANGE[1] + 1))
        
        n_edge_types = n_prov + n_ref + n_struct + n_cooc
        
        # Enforce MAX_EDGE_TYPES ceiling
        if n_edge_types > MAX_EDGE_TYPES:
            excess = n_edge_types - MAX_EDGE_TYPES
            trim_cooc = min(excess, n_cooc)
            n_cooc -= trim_cooc
            excess -= trim_cooc
            if excess > 0:
                n_ref = max(2, n_ref - excess)
            n_edge_types = n_prov + n_ref + n_struct + n_cooc
        
        # --- Build edge type specifications ---
        edge_specs = []
        edge_idx = 0
        
        # Helper to get node type indices for a layer
        source_types = [i for i, l in enumerate(node_layer_assignment) if l == LAYER_SOURCE]
        claim_types = [i for i, l in enumerate(node_layer_assignment) if l == LAYER_CLAIM]
        entity_types = [i for i, l in enumerate(node_layer_assignment) if l == LAYER_ENTITY]
        auxiliary_types = [i for i, l in enumerate(node_layer_assignment) if l == LAYER_AUXILIARY]
        
        # Provenance edges: claim -> source (or claim -> claim for multi-hop)
        for _ in range(n_prov):
            # Decide if this provenance edge connects claim->source or claim->claim
            if rng.random() < 0.6 or len(edge_specs) == 0:
                # claim -> source (most common)
                src_types = self._sample_subset(rng, claim_types, min_size=1)
                tgt_types = self._sample_subset(rng, source_types, min_size=1)
                src_layers = [LAYER_CLAIM]
                tgt_layers = [LAYER_SOURCE]
            else:
                # claim -> claim (multi-hop provenance)
                src_types = self._sample_subset(rng, claim_types, min_size=1)
                tgt_types = self._sample_subset(rng, claim_types, min_size=1)
                src_layers = [LAYER_CLAIM]
                tgt_layers = [LAYER_CLAIM]
            
            edge_specs.append(EdgeTypeSpec(
                index=edge_idx,
                category=EDGE_CAT_PROVENANCE,
                valid_source_layers=src_layers,
                valid_target_layers=tgt_layers,
                valid_source_types=src_types,
                valid_target_types=tgt_types,
                directed=True,
            ))
            edge_idx += 1
        
        # Ensure at least one claim->source provenance edge exists
        has_claim_to_source = any(
            e.category == EDGE_CAT_PROVENANCE and 
            LAYER_SOURCE in e.valid_target_layers
            for e in edge_specs
        )
        if not has_claim_to_source:
            # Override the first provenance edge
            edge_specs[0] = EdgeTypeSpec(
                index=0,
                category=EDGE_CAT_PROVENANCE,
                valid_source_layers=[LAYER_CLAIM],
                valid_target_layers=[LAYER_SOURCE],
                valid_source_types=self._sample_subset(rng, claim_types, min_size=1),
                valid_target_types=self._sample_subset(rng, source_types, min_size=1),
                directed=True,
            )
        
        # Reference edges: claim -> entity
        for _ in range(n_ref):
            src_types = self._sample_subset(rng, claim_types, min_size=1)
            tgt_types = self._sample_subset(rng, entity_types, min_size=1)
            
            edge_specs.append(EdgeTypeSpec(
                index=edge_idx,
                category=EDGE_CAT_REFERENCE,
                valid_source_layers=[LAYER_CLAIM],
                valid_target_layers=[LAYER_ENTITY],
                valid_source_types=src_types,
                valid_target_types=tgt_types,
                directed=True,
            ))
            edge_idx += 1
        
        # Structural edges: various connections (temporal, supersedes, etc.)
        for _ in range(n_struct):
            struct_pattern = int(rng.integers(0, 3))
            
            if struct_pattern == 0:
                # Temporal: source/claim -> auxiliary (time period)
                if len(auxiliary_types) > 0:
                    src_layers_opt = [LAYER_SOURCE, LAYER_CLAIM]
                    src_layer = [src_layers_opt[int(rng.integers(0, 2))]]
                    src_t = self._sample_subset(
                        rng,
                        [i for i, l in enumerate(node_layer_assignment) if l == src_layer[0]],
                        min_size=1
                    )
                    tgt_t = self._sample_subset(rng, auxiliary_types, min_size=1)
                    tgt_layer = [LAYER_AUXILIARY]
                else:
                    # No auxiliary types; connect source -> source (supersedes)
                    src_layer = [LAYER_SOURCE]
                    tgt_layer = [LAYER_SOURCE]
                    src_t = self._sample_subset(rng, source_types, min_size=1)
                    tgt_t = self._sample_subset(rng, source_types, min_size=1)
            elif struct_pattern == 1:
                # Cross-layer structural: source -> source (supersedes)
                src_layer = [LAYER_SOURCE]
                tgt_layer = [LAYER_SOURCE]
                src_t = self._sample_subset(rng, source_types, min_size=1)
                tgt_t = self._sample_subset(rng, source_types, min_size=1)
            else:
                # Claim-level structural: claim -> claim (corroborates)
                src_layer = [LAYER_CLAIM]
                tgt_layer = [LAYER_CLAIM]
                src_t = self._sample_subset(rng, claim_types, min_size=1)
                tgt_t = self._sample_subset(rng, claim_types, min_size=1)
            
            edge_specs.append(EdgeTypeSpec(
                index=edge_idx,
                category=EDGE_CAT_STRUCTURAL,
                valid_source_layers=src_layer,
                valid_target_layers=tgt_layer,
                valid_source_types=src_t,
                valid_target_types=tgt_t,
                directed=True,
            ))
            edge_idx += 1
        
        # Co-occurrence edges: entity <-> entity (undirected)
        for _ in range(n_cooc):
            # Pick two (possibly overlapping) subsets of entity types
            src_types = self._sample_subset(rng, entity_types, min_size=1)
            tgt_types = self._sample_subset(rng, entity_types, min_size=1)
            
            edge_specs.append(EdgeTypeSpec(
                index=edge_idx,
                category=EDGE_CAT_COOCCURRENCE,
                valid_source_layers=[LAYER_ENTITY],
                valid_target_layers=[LAYER_ENTITY],
                valid_source_types=src_types,
                valid_target_types=tgt_types,
                directed=False,
            ))
            edge_idx += 1
        
        return SchemaDescriptor(
            n_source_types=n_source,
            n_claim_types=n_claim,
            n_entity_types=n_entity,
            n_auxiliary_types=n_auxiliary,
            n_node_types=n_node_types,
            n_edge_types=n_edge_types,
            node_layer_assignment=node_layer_assignment,
            edge_specs=edge_specs,
            n_provenance_edges=n_prov,
            n_reference_edges=n_ref,
            n_structural_edges=n_struct,
            n_cooccurrence_edges=n_cooc,
            seed=seed,
        )
    
    def sample_batch(self, n: int) -> List[SchemaDescriptor]:
        """Generate n schemas with sequential derived seeds."""
        return [self.sample() for _ in range(n)]
    
    @staticmethod
    def _sample_subset(rng: np.random.Generator, 
                       items: List[int], 
                       min_size: int = 1) -> List[int]:
        """
        Sample a random non-empty subset from a list.
        Returns at least min_size items (or all items if fewer exist).
        """
        if len(items) <= min_size:
            return list(items)
        
        size = int(rng.integers(min_size, len(items) + 1))
        indices = rng.choice(len(items), size=size, replace=False)
        return [items[i] for i in sorted(indices)]


def validate_schema(schema: SchemaDescriptor) -> List[str]:
    """
    Validate that a schema satisfies all structural invariants.
    
    Returns:
        List of error messages. Empty list = valid schema.
    """
    errors = []
    
    # Invariant: at least 1 source type
    if schema.n_source_types < 1:
        errors.append(f"Must have >= 1 source type, got {schema.n_source_types}")
    
    # Invariant: at least 1 claim type
    if schema.n_claim_types < 1:
        errors.append(f"Must have >= 1 claim type, got {schema.n_claim_types}")
    
    # Invariant: at least 2 entity types
    if schema.n_entity_types < 2:
        errors.append(f"Must have >= 2 entity types, got {schema.n_entity_types}")
    
    # Invariant: at least 1 provenance edge
    if schema.n_provenance_edges < 1:
        errors.append(f"Must have >= 1 provenance edge type, got {schema.n_provenance_edges}")
    
    # Invariant: at least 1 reference edge
    if schema.n_reference_edges < 1:
        errors.append(f"Must have >= 1 reference edge type, got {schema.n_reference_edges}")
    
    # Invariant: at least 1 structural edge
    if schema.n_structural_edges < 1:
        errors.append(f"Must have >= 1 structural edge type, got {schema.n_structural_edges}")
    
    # Invariant: at least one provenance edge connects claim -> source
    has_claim_to_source = any(
        e.category == EDGE_CAT_PROVENANCE and LAYER_SOURCE in e.valid_target_layers
        for e in schema.edge_specs
    )
    if not has_claim_to_source:
        errors.append("No provenance edge type connects claims to sources")
    
    # Invariant: at least one reference edge connects claim -> entity
    has_claim_to_entity = any(
        e.category == EDGE_CAT_REFERENCE and LAYER_ENTITY in e.valid_target_layers
        for e in schema.edge_specs
    )
    if not has_claim_to_entity:
        errors.append("No reference edge type connects claims to entities")
    
    # Consistency: n_node_types matches layer assignment length
    if len(schema.node_layer_assignment) != schema.n_node_types:
        errors.append(
            f"Layer assignment length ({len(schema.node_layer_assignment)}) "
            f"!= n_node_types ({schema.n_node_types})"
        )
    
    # Consistency: n_edge_types matches edge_specs length
    if len(schema.edge_specs) != schema.n_edge_types:
        errors.append(
            f"Edge specs length ({len(schema.edge_specs)}) "
            f"!= n_edge_types ({schema.n_edge_types})"
        )
    
    # Consistency: edge indices are sequential
    for i, e in enumerate(schema.edge_specs):
        if e.index != i:
            errors.append(f"Edge spec {i} has index {e.index}, expected {i}")
    
    # Consistency: edge source/target types exist in the schema
    for e in schema.edge_specs:
        for t in e.valid_source_types:
            if t >= schema.n_node_types:
                errors.append(f"Edge {e.index}: source type {t} >= n_node_types")
        for t in e.valid_target_types:
            if t >= schema.n_node_types:
                errors.append(f"Edge {e.index}: target type {t} >= n_node_types")
    
    # Size limits
    if schema.n_node_types > MAX_NODE_TYPES:
        errors.append(f"n_node_types ({schema.n_node_types}) > MAX ({MAX_NODE_TYPES})")
    if schema.n_edge_types > MAX_EDGE_TYPES:
        errors.append(f"n_edge_types ({schema.n_edge_types}) > MAX ({MAX_EDGE_TYPES})")
    
    return errors


# --- Quick self-test ---
if __name__ == "__main__":
    sampler = SchemaSampler(master_seed=42)
    
    print("Generating 20 schemas for inspection...\n")
    
    all_valid = True
    type_counts = []
    edge_counts = []
    
    for i in range(20):
        schema = sampler.sample()
        errors = validate_schema(schema)
        
        if errors:
            print(f"Schema {i} INVALID:")
            for e in errors:
                print(f"  ERROR: {e}")
            all_valid = False
        
        type_counts.append(schema.n_node_types)
        edge_counts.append(schema.n_edge_types)
    
    print(f"All 20 schemas valid: {all_valid}")
    print(f"Node type counts: min={min(type_counts)}, max={max(type_counts)}, "
          f"mean={np.mean(type_counts):.1f}")
    print(f"Edge type counts: min={min(edge_counts)}, max={max(edge_counts)}, "
          f"mean={np.mean(edge_counts):.1f}")
    
    # Print a few examples
    print("\n" + "="*60)
    print("SAMPLE SCHEMAS")
    print("="*60)
    
    sampler2 = SchemaSampler(master_seed=99)
    for i in range(3):
        schema = sampler2.sample()
        print(f"\n{schema.summary()}")
    
    # Test deterministic reproduction
    print("\n" + "="*60)
    print("DETERMINISM TEST")
    print("="*60)
    s1 = SchemaSampler(master_seed=42).sample(seed=12345)
    s2 = SchemaSampler(master_seed=99).sample(seed=12345)  # different master, same schema seed
    assert s1.n_node_types == s2.n_node_types
    assert s1.n_edge_types == s2.n_edge_types
    assert s1.node_layer_assignment == s2.node_layer_assignment
    print("Deterministic reproduction confirmed: same seed -> same schema")
    
    # Test tensor conversion
    print("\n" + "="*60)
    print("TENSOR CONVERSION TEST")
    print("="*60)
    td = s1.to_tensor_dict()
    for k, v in td.items():
        print(f"  {k:30s} shape={str(v.shape):15s} dtype={v.dtype}")
    
    print("\nAll tests passed.")
