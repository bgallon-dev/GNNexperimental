# KGR v1.0 — Frozen Deployed Model (2026-07-07)

This folder is a **frozen, self-contained copy** of every artifact in the
deployed KGR system as of 2026-07-07. Nothing in here is a working file:
do not retrain into, overwrite, or "fix" anything in this tree. All files
are set read-only and hashed in `MANIFEST.sha256`.

Verify integrity at any time (Git Bash, from this directory):

```bash
sha256sum -c MANIFEST.sha256 --quiet && echo FROZEN-OK
```

## What this is

The per-task-family **router** system described in
[Docs/PROJECT_HANDOFF.md](../../Docs/PROJECT_HANDOFF.md) §1: a frozen
hyperbolic encoder + QueryToBall head, per-task reranker weights for the two
recipes (v3.2-damped / v3.3-blend), and the router verdicts that choose a
recipe per task. Synthetic ndcg@10 0.371 (retriever 0.282, zero deployed
regressions); real-temporal routed 0.795 (anchor-BFS 0.650).

## Inventory

| path | contents | provenance (runs/) |
|---|---|---|
| `encoder_baseline/` | locked v3.1 encoder (`encoder.pt`, SHA `ed8139dc8209...`), synthetic `query_encoder.pt`, `baseline_manifest.json` (SHA + noise floor + gate rule), frozen Eval A–E `*_baseline.json`, `manifold_index.npz`, `PHASE1_FINDINGS.md` (full operational log) | `v3.1-baseline-hyp-h128-l4-seed1` |
| `real_head/` | real-trained Stage-B qh1 head (task 2), SHA-identical encoder copy, real-val manifold index, cmp JSONs (0.242 frozen → 0.558 real-trained) | `v3.1-real-head-hyp-h128-l4-seed0` |
| `reranker_synthetic/v32/`, `v33/` | per-task×seed `reranker.pt` (tasks 0–5 × seeds 0–2) + sweep results, both recipes, synthetic domain | `sweep_reranker_v32`, `sweep_reranker_v33` |
| `reranker_real/v32/`, `v33/` | per-seed `reranker.pt` (task 2) + sweep results, both recipes, real domain | `sweep_reranker_real_v32`, `sweep_reranker_real_v33` |
| `router/synthetic/`, `router/real/` | `router_results.json` — validation-gated per-task recipe choice (the deployed verdict tables) | `reranker_router`, `reranker_router_real` |
| `MANIFEST.sha256` | SHA256 of every file in this tree | — |

## How to load

Weight-loading / inference code is documented in
[HANDOFF.md](../../HANDOFF.md) (source of truth) — the run-dir layout here is
identical to the original `runs/` dirs, so any loader that takes a run dir
(e.g. `retrieval_ops.load_query_encoder(run_dir)`) accepts these paths
directly:

- synthetic retrieval head: `frozen/kgr-v1.0-2026-07-07/encoder_baseline`
- real-temporal retrieval head: `frozen/kgr-v1.0-2026-07-07/real_head`

The manifold index must come from the SHA-identical frozen encoder — true for
both dirs above (verified: all encoder copies hash to `ed8139dc8209...`).

## Known limitations of this model (documented, do not re-litigate)

- Corpus-wide (pool) retrieval is near-random — it is a **reranker**; give it
  candidates (see PROJECT_HANDOFF.md §6, the open thread).
- Task-3 multi-hop and task-1 entity resolution are standing limitations.
- `--stage-b-head bilinear` is research-only (refuted in transfer); the
  temporal-aux flag is the validated lever.
