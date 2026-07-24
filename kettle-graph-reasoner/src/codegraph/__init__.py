"""Code-graph harness for KGR.

Ingests a `kgr_codegraph_odin` export (nodes.jsonl / edges.jsonl /
training_cases.jsonl), encodes it into the tier1 schema the frozen KGR
encoder was trained on, trains a per-task `QueryToBall` head, and reports
ranking metrics. The graph encoder is reused frozen — this is a
schema-portability probe on a code call graph, not a retrain.
"""
