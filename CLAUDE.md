# Project: Kettle Graph Reasoner (KGR)

## What This Is

A small (500K–2M parameter) graph neural network that serves as a **structural reasoning layer** between a Neo4j knowledge graph and an LLM. It replaces heuristic subgraph retrieval with learned, query-conditioned relevance scoring. The model receives a query and a graph neighborhood, reasons over the structure, and outputs a **ranked, pruned subgraph** — node/edge relevance scores — that the LLM then receives as high-signal context.

This is a **research prototype**, not a production system. The goal is a proof of concept demonstrating that graph-native structural reasoning outperforms flat-text serialization on structural tasks (provenance traversal, entity resolution, temporal scoping, multi-hop inference). Positive results at any margin constitute success.

## Core Architectural Commitments

These are non-negotiable design constraints. Do not suggest alternatives that abandon these:

1. **Hyperbolic geometry.** Node embeddings live in the Poincaré ball, not Euclidean space. Message passing uses Möbius operations (Möbius addition, exponential/logarithmic maps). Positional encodings derive from hyperbolic distance, not sequential position. Use `geoopt` for Riemannian optimization primitives.

2. **Edge-typed heterogeneous attention.** Each edge type (MENTIONS, DERIVED_FROM, TEMPORAL_SCOPE, PUBLISHED_IN, etc.) gets its own learned attention parameters. Edges are first-class representational objects, not uniform conduits. The model must distinguish a provenance relationship from a co-mention relationship at the architectural level.

3. **Schema-portable.** The model receives a schema descriptor (node types, edge types, hierarchy specification) alongside the graph. It must NOT have hardcoded type embeddings for specific domains. The learned competence is "graph reasoning patterns," not domain content. A model trained on diverse synthetic graphs should generalize to unseen real-world schemas without retraining.

4. **Output is structural, not linguistic.** The model outputs relevance scores per node and per edge, plus optionally a subgraph boundary mask. It never produces natural language. No language modeling head, no vocabulary embeddings. This is a ranking/scoring model.

5. **Tiny by design.** Target 500K–2M parameters. This is a feature, not a limitation. The model must be fast enough to run as a preprocessing step on every query. If a design choice increases parameter count significantly, it needs strong justification.

## What This Project Is NOT

- **Not a GraphRAG reimplementation.** Do not reproduce Microsoft GraphRAG, LlamaIndex KG mode, or any existing graph-retrieval pipeline. We are building a component those systems don't have.
- **Not a knowledge graph embedding model.** We are not doing TransE/DistMult/RotatE-style link prediction. The task is query-conditioned subgraph relevance scoring, not static knowledge graph completion.
- **Not a standard GNN benchmark exercise.** Do not default to Cora/Citeseer/PubMed node classification. Our evaluation tasks are custom: provenance chain traversal, entity resolution across subgraphs, temporal scope filtering, multi-hop structural inference.
- **Not an LLM fine-tune.** The language model is a downstream consumer. This project does not touch the LLM.

## Anti-Patterns — Do Not Do These

- **Do not flatten graphs into token sequences.** If you find yourself serializing graph structure into a string for a transformer to read, you are solving the wrong problem. That is the exact bottleneck we are replacing.
- **Do not default to standard GCN/GAT/GraphSAGE.** These are Euclidean, homogeneous, node-centric baselines. They are the *control condition*, not the experimental condition. Implement them only as baselines for comparison.
- **Do not use off-the-shelf PyG models without modification.** PyTorch Geometric provides useful primitives (data loaders, batching, sparse ops), but the message-passing layers must be custom — hyperbolic, edge-typed, schema-conditioned. Using `GCNConv` or `GATConv` directly is implementing the baseline, not the experiment.
- **Do not train on a single real-world graph.** Training exclusively on Turnbull or newspaper data is memorization. The training data must be synthetically generated across diverse schemas. Real-world graphs are for evaluation only.
- **Do not optimize for benchmark leaderboards.** We are not trying to beat SOTA on FB15k-237 or WN18RR. We are testing a specific hypothesis about representation alignment on our own evaluation suite.
- **Do not add a language modeling head.** If you're importing `transformers` or adding token embeddings, you've drifted off-target.

## Research Hypothesis

**A small GNN operating in hyperbolic space with edge-typed heterogeneous attention, trained on structurally diverse synthetic knowledge graphs, will produce higher-quality query-conditioned subgraph selections than (a) heuristic template retrieval and (b) an equivalently-sized Euclidean homogeneous GNN, when evaluated on structural reasoning tasks over real-world archival knowledge graphs.**

The independent variable is the representation geometry and attention structure. The dependent variable is downstream retrieval quality (precision, recall, and LLM answer quality when consuming the model's output vs. baselines).

## Technical Stack

- **Python 3.10+** — prototype language (Colab compatibility)
- **PyTorch** — tensor operations, autograd
- **PyTorch Geometric (PyG)** — graph data structures, batching, sparse utilities (NOT pre-built convolution layers for the experimental model)
- **geoopt** — Riemannian manifold operations, Poincaré ball math, Riemannian Adam optimizer
- **Neo4j / neo4j Python driver** — real-world graph access for evaluation
- **NetworkX** — synthetic graph generation
- **Weights & Biases (wandb)** or **TensorBoard** — experiment tracking

## Project Structure

```
kettle-graph-reasoner/
├── CLAUDE.md
├── src/
│   ├── models/
│   │   ├── hyperbolic_gnn.py           # Experimental model
│   │   ├── euclidean_baseline.py       # GCN/GAT baseline
│   │   ├── euclidean_plus_baseline.py  # Euclidean + edge-typed attn baseline
│   │   └── layers/
│   │       ├── poincare_ops.py     # Möbius math, exp/log maps
│   │       ├── hyp_message_pass.py # Hyperbolic message passing
│   │       ├── edge_attention.py   # Edge-typed attention mechanism
│   │       └── schema_encoder.py   # Schema descriptor encoding
│   ├── data/
│   │   ├── corpus_builder.py       # Build synthetic training corpus
│   │   ├── corpus_dataset.py       # PyG dataset wrapper
│   │   ├── schema_sampler.py       # Schema archetype sampling
│   │   ├── graph_builder.py        # Graph topology generation
│   │   ├── feature_encoder.py      # Node/edge feature encoding
│   │   └── task_generator.py       # Evaluation task construction
│   └── training/
│       ├── train.py                # Training entrypoint
│       ├── loss.py                 # Ranking / relevance losses
│       └── metrics.py              # Retrieval quality metrics
├── configs/
│   └── experiment.yaml
├── runs/                           # Training run outputs (gitignored)
└── tests/
    ├── test_poincare_ops.py
    ├── test_message_passing.py
    ├── test_edge_attention.py
    ├── test_schema_encoding.py
    ├── test_euclidean_baseline.py
    └── test_full_model.py

# Sibling to kettle-graph-reasoner/:
# 01_graph_properties.ipynb           # Real-graph statistics notebook
```

Not yet implemented: `src/eval/` suite (structural-task evaluation, pipeline A/B), Neo4j loader for real-graph evaluation, and `src/utils/` (visualization, config helpers). Training currently runs on synthetic corpora only; real-graph evaluation on Turnbull/newspaper data is still to come.

## Development Workflow

1. **Measure first.** Before building anything, measure the real graph properties: degree distribution, Gromov hyperbolicity, provenance chain depth, branching factor. These numbers determine whether hyperbolic geometry is justified and constrain the architecture.

2. **Build the math layer.** Implement and test Poincaré ball operations in isolation. These must be numerically stable (clamp norms, handle boundary cases). Verify against `geoopt` reference implementations.

3. **Build the synthetic data generator.** This is as important as the model. Generate diverse schemas, diverse topologies, diverse structural reasoning tasks. The training data quality determines everything.

4. **Build the experimental model.** Hyperbolic message passing + edge-typed attention + schema conditioning. Start minimal, add complexity only when justified by evaluation.

5. **Build the baselines.** Standard GCN/GAT on the same data. Heuristic template retrieval on the same evaluation tasks. These exist only for comparison.

6. **Evaluate on real graphs.** Load Turnbull and newspaper graphs from Neo4j. Run the evaluation suite. Compare experimental vs. baselines.

## When You're Stuck

- If a standard approach seems like the obvious solution, **explain why it's insufficient for our specific constraints** before implementing it.
- If the hyperbolic math is numerically unstable, **don't fall back to Euclidean** — debug the numerics. Clamp norms, use the right epsilon values, check the curvature parameter.
- If you're unsure whether something violates the architectural commitments, **ask** rather than defaulting to the conventional approach.
- **Novelty is expected.** There may not be a tutorial for what we're building. That's the point. Compose from primitives, cite papers, implement from equations — don't search for a plug-and-play solution.

## Key References

- Poincaré ball operations: Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations" (2017)
- Hyperbolic GNNs: Liu et al., "Hyperbolic Graph Neural Networks" (2019)
- Hyperbolic KG reasoning: Liu, "HyperKGR" (EMNLP 2025)
- Geometry-task alignment: Katsman & Gilbert (2025); "Hyperbolic GNNs Under the Microscope" (2026)
- Edge-centric attention: "An end-to-end attention-based approach for learning on graphs" (Nature Communications, 2025)
- Heterogeneous graph memory: Yu et al., "HMT" (Scientific Reports, 2025)
- Temporal KG in hyperbolic space: Li et al., "HyGNet" (LREC 2024)
