"""
Corpus Builder for the Kettle Graph Reasoner synthetic data pipeline.

Orchestrates the full pipeline: schema sampling -> graph building ->
task generation -> feature encoding -> serialization.

Produces a training corpus of .pt files (PyTorch Geometric format)
with corpus-level metadata for reproducibility.

Reference: KGR Synthetic Generator Spec v0.1, Sections 6-7.

Usage:
    python corpus_builder.py --tier 1 --output ./corpus --seed 42
    python corpus_builder.py --tier 2 --output ./corpus --seed 42
    python corpus_builder.py --tier 3 --output ./corpus --seed 42
"""

import os
import json
import time
import argparse
import numpy as np
from typing import Dict, List, Optional
from dataclasses import asdict

from schema_sampler import SchemaSampler, SchemaDescriptor, validate_schema
from graph_builder import GraphBuilder, BuilderConfig, SyntheticGraph
from feature_encoder import (
    encode_nodes, encode_edges, encode_query,
    NODE_FEAT_DIM, EDGE_FEAT_DIM, QUERY_FEAT_DIM,
)
from task_generator import TaskGenerator, TaskExample

# Tier configurations (from spec Section 3.4)
TIER_CONFIGS = {
    1: {"n_graphs": 100, "node_range": (50, 500), "label": "Pipeline validation"},
    2: {"n_graphs": 1000, "node_range": (100, 2000), "label": "Initial training"},
    3: {"n_graphs": 10000, "node_range": (100, 5000), "label": "Full experimental run"},
}


def build_single_graph(schema: SchemaDescriptor, graph_seed: int,
                        target_nodes: int, task_seed: int) -> Optional[Dict]:
    """
    Build a single graph with all features and tasks.
    
    Returns a dictionary ready for serialization, or None if validation fails.
    """
    # Build graph
    config = BuilderConfig(target_nodes=target_nodes)
    builder = GraphBuilder(config)
    graph = builder.build(schema, seed=graph_seed)
    
    # Encode features
    node_features, id_to_row = encode_nodes(graph)
    edge_index, edge_attr = encode_edges(graph, id_to_row)
    schema_tensor = schema.to_tensor_dict()
    
    # Generate tasks
    task_gen = TaskGenerator(seed=task_seed)
    tasks = task_gen.generate_all_tasks(graph, id_to_row)
    
    # Serialize tasks
    task_dicts = []
    for t in tasks:
        td = {
            "task_type": t.task_type,
            "anchor_node": t.anchor_node,
            "anchor_row": id_to_row.get(t.anchor_node, 0),
            "labels": t.labels,
            "max_hops": t.max_hops,
        }
        if t.temporal_window is not None:
            td["temporal_window"] = np.array(t.temporal_window, dtype=np.float32)
        if t.er_pairs is not None:
            td["er_pairs"] = np.array(t.er_pairs, dtype=np.int64)
        
        td["query_features"] = encode_query(
            task_type=t.task_type,
            anchor_row=td["anchor_row"],
            temporal_window=t.temporal_window,
            max_hops=t.max_hops,
        )
        
        task_dicts.append(td)
    
    # Build duplicate pairs array
    dup_pairs = np.array(
        [(id_to_row.get(a, 0), id_to_row.get(b, 0), tier)
         for a, b, tier in graph.duplicate_pairs],
        dtype=np.int64
    ) if graph.duplicate_pairs else np.zeros((0, 3), dtype=np.int64)
    
    return {
        "x": node_features,                # (N, 32)
        "edge_index": edge_index,           # (2, E)
        "edge_attr": edge_attr,             # (E, 30)
        "schema": schema_tensor,            # dict of arrays
        "tasks": task_dicts,                # list of task dicts
        "duplicate_pairs": dup_pairs,       # (D, 3)
        "seed": graph_seed,
        "schema_seed": schema.seed,
        "n_nodes": graph.n_nodes,
        "n_edges": graph.n_edges,
        "n_tasks": len(task_dicts),
    }


def build_corpus(tier: int, output_dir: str, master_seed: int = 42,
                  verbose: bool = True):
    """
    Build an entire corpus for a given tier.
    
    Args:
        tier: 1, 2, or 3
        output_dir: root directory for output
        master_seed: deterministic seed for everything
        verbose: print progress
    """
    if tier not in TIER_CONFIGS:
        raise ValueError(f"Unknown tier {tier}. Must be 1, 2, or 3.")
    
    config = TIER_CONFIGS[tier]
    n_graphs = config["n_graphs"]
    node_lo, node_hi = config["node_range"]
    
    tier_dir = os.path.join(output_dir, f"tier{tier}")
    os.makedirs(tier_dir, exist_ok=True)
    
    if verbose:
        print(f"Building Tier {tier} corpus: {n_graphs} graphs")
        print(f"  Node range: [{node_lo}, {node_hi}]")
        print(f"  Output: {tier_dir}")
        print(f"  Master seed: {master_seed}")
        print()
    
    # Initialize samplers with deterministic seeds
    rng = np.random.default_rng(master_seed)
    schema_sampler = SchemaSampler(master_seed=int(rng.integers(0, 2**31)))
    
    # Pre-generate all seeds for reproducibility
    graph_seeds = rng.integers(0, 2**31, size=n_graphs).tolist()
    task_seeds = rng.integers(0, 2**31, size=n_graphs).tolist()
    target_sizes = np.exp(
        rng.uniform(np.log(node_lo), np.log(node_hi), size=n_graphs)
    ).astype(int).tolist()  # log-uniform distribution
    
    # Tracking
    total_nodes = 0
    total_edges = 0
    total_tasks = 0
    total_dups = 0
    schema_seeds = []
    start_time = time.time()
    errors = 0
    
    for i in range(n_graphs):
        # Generate schema
        schema = schema_sampler.sample()
        validation_errors = validate_schema(schema)
        if validation_errors:
            if verbose:
                print(f"  [{i:5d}] Schema validation failed: {validation_errors}")
            errors += 1
            continue
        
        # Build graph + tasks
        try:
            data = build_single_graph(
                schema=schema,
                graph_seed=graph_seeds[i],
                target_nodes=target_sizes[i],
                task_seed=task_seeds[i],
            )
        except Exception as e:
            if verbose:
                print(f"  [{i:5d}] Build failed: {e}")
            errors += 1
            continue
        
        if data is None:
            errors += 1
            continue
        
        # Save as numpy archive (.npz) — lightweight, no torch dependency
        filepath = os.path.join(tier_dir, f"graph_{i:06d}.npz")
        
        # Flatten for npz storage
        save_dict = {
            "x": data["x"],
            "edge_index": data["edge_index"],
            "edge_attr": data["edge_attr"],
            "duplicate_pairs": data["duplicate_pairs"],
            "seed": np.array(data["seed"]),
            "schema_seed": np.array(data["schema_seed"]),
        }
        
        # Schema tensors
        for k, v in data["schema"].items():
            save_dict[f"schema_{k}"] = v
        
        # Tasks
        save_dict["n_tasks"] = np.array(len(data["tasks"]))
        for j, task in enumerate(data["tasks"]):
            save_dict[f"task_{j}_type"] = np.array(task["task_type"])
            save_dict[f"task_{j}_anchor_row"] = np.array(task["anchor_row"])
            save_dict[f"task_{j}_labels"] = task["labels"]
            save_dict[f"task_{j}_query"] = task["query_features"]
            save_dict[f"task_{j}_max_hops"] = np.array(task["max_hops"])
            if "temporal_window" in task:
                save_dict[f"task_{j}_temporal"] = task["temporal_window"]
            if "er_pairs" in task:
                save_dict[f"task_{j}_er_pairs"] = task["er_pairs"]
        
        np.savez_compressed(filepath, **save_dict)
        
        # Track statistics
        total_nodes += data["n_nodes"]
        total_edges += data["n_edges"]
        total_tasks += data["n_tasks"]
        total_dups += len(data["duplicate_pairs"])
        schema_seeds.append(data["schema_seed"])
        
        if verbose and (i + 1) % max(1, n_graphs // 20) == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_graphs - i - 1) / rate
            print(f"  [{i+1:5d}/{n_graphs}] "
                  f"nodes={data['n_nodes']:4d} edges={data['n_edges']:4d} "
                  f"tasks={data['n_tasks']:2d} "
                  f"({rate:.1f} graphs/s, ETA {eta:.0f}s)")
    
    elapsed = time.time() - start_time
    
    # Write metadata
    metadata = {
        "tier": tier,
        "n_graphs": n_graphs,
        "n_generated": n_graphs - errors,
        "n_errors": errors,
        "master_seed": master_seed,
        "node_range": [node_lo, node_hi],
        "total_nodes": int(total_nodes),
        "total_edges": int(total_edges),
        "total_tasks": int(total_tasks),
        "total_duplicate_pairs": int(total_dups),
        "avg_nodes": float(total_nodes / max(n_graphs - errors, 1)),
        "avg_edges": float(total_edges / max(n_graphs - errors, 1)),
        "avg_tasks": float(total_tasks / max(n_graphs - errors, 1)),
        "n_unique_schemas": len(set(schema_seeds)),
        "generation_time_seconds": round(elapsed, 2),
        "graphs_per_second": round((n_graphs - errors) / max(elapsed, 0.01), 2),
        "graph_seeds": graph_seeds,
        "feature_dims": {
            "node": NODE_FEAT_DIM,
            "edge": EDGE_FEAT_DIM,
            "query": QUERY_FEAT_DIM,
        },
    }
    
    meta_path = os.path.join(tier_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"\nCorpus generation complete.")
        print(f"  Generated: {n_graphs - errors}/{n_graphs} graphs")
        print(f"  Total nodes: {total_nodes:,}")
        print(f"  Total edges: {total_edges:,}")
        print(f"  Total tasks: {total_tasks:,}")
        print(f"  Total duplicate pairs: {total_dups:,}")
        print(f"  Unique schemas: {len(set(schema_seeds))}")
        print(f"  Time: {elapsed:.1f}s ({(n_graphs-errors)/max(elapsed,0.01):.1f} graphs/s)")
        print(f"  Output: {tier_dir}")
        print(f"  Metadata: {meta_path}")
    
    return metadata


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KGR synthetic training corpus")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                        help="Corpus tier (1=100 graphs, 2=1000, 3=10000)")
    parser.add_argument("--output", type=str, default="./corpus",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    
    args = parser.parse_args()
    
    metadata = build_corpus(
        tier=args.tier,
        output_dir=args.output,
        master_seed=args.seed,
        verbose=not args.quiet,
    )
