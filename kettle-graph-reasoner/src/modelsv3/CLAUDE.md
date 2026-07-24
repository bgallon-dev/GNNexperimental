# `src/modelsv3/` — embedding-first, contrastive KGR

> **[SCOPE STAMP - 2026-05-17]** This document is scoped to the v3 embedding-first regime guide; the Success-Criteria section is partly falsified by Phase-2+ (see the master / PHASE1_FINDINGS.md). For current deployed state, the honest arc, the document map, and the open thread, the master is **[Docs/PROJECT_HANDOFF.md](../../Docs/PROJECT_HANDOFF.md)** (source of truth for current status). Content below is preserved as-is and may be stale where the master says so.

Third-generation redesign. v3 is a **regime shift** from v1/v2, not an
incremental model improvement: it replaces score-first supervised
training with embedding-first self-supervised training. See
[../models/CLAUDE.md](../models/CLAUDE.md) for v1 and
[../modelsv2/CLAUDE.md](../modelsv2/CLAUDE.md) for v2.

## What changed

| Dimension | v1 / v2 | v3 |
| --- | --- | --- |
| Primary output | per-node sigmoid score | per-node Poincaré-ball embedding |
| Training loss | BCE/MSE on scalar labels | InfoNCE on embedding distance (stage A) + pairwise ranking (stage B) |
| Query entry | concatenated into scoring head | separate `QueryToBall` head, mapped to a ball point |
| Task conditioning | per-task scoring branch | query encoder is task-scoped; graph encoder is unconditional |
| Positive pairs | N/A (labels are the supervision) | edge + same-label-different-features, on-the-fly |

## Contents

- [hyperbolic_gnnV3.py](hyperbolic_gnnV3.py) — `KettleGraphReasonerV3`. Forked from v2; scoring heads + query path stripped. Produces `KGREmbeddingOutput { node_embeddings, edge_type_embeddings, per_round_embeddings }`.
- [euclidean_v3.py](euclidean_v3.py) — `EuclideanReasonerV3`. **Mandatory baseline** for the geometric claim within the contrastive regime. Same architecture modulo geometry: Euclidean input projection, `EdgeTypedAttention(euclidean=True)`, `EuclideanMessagePassing` (imported from v1), Euclidean depth attention.
- [query_encoder.py](query_encoder.py) — `QueryToBall`. Two-layer MLP mapping the per-task query vector to a point on the same Poincaré ball (or Euclidean space, with `euclidean=True`). Same small-gain Xavier + `tangent_scale` recipe as the graph encoder's input projection.
- [contrastive.py](contrastive.py) — `PositiveSampler` + `poincare_infonce`. Stage-A training signal: edge positives (orthogonal to `x`) mixed with same-label-different-features positives (forces the encoder to learn node-type abstraction from graph context rather than feature cosine). Negatives: intra-graph, k-hop neighbors excluded.
- [ranking.py](ranking.py) — `pairwise_ranking_loss` (default) and `listwise_ranking_loss` (fallback). Stage-B loss. **Never MSE** — MSE on scalar distances reintroduces the mean-collapse failure mode stage A was built to escape.
- [distance_scoring.py](distance_scoring.py) — `score_from_embeddings`. Eval-time bridge from trained embeddings to [`../training/metrics.py`](../training/metrics.py). Hyperbolic and Euclidean variants share the same call signature.
- [intrinsic_eval.py](intrinsic_eval.py) — silhouette, nn-edge-precision, nn-label-purity. Intrinsic embedding-quality metrics independent of the task head.
- [three_seed_comparison_v3.py](three_seed_comparison_v3.py) — multi-arm comparison harness. Runs v3_hyp vs v3_euc across seeds; optionally folds in pre-computed v1/v2 summary.json files via `--extra-baseline`.

Training entrypoint lives in the shared training folder:
- [../training/train_v3.py](../training/train_v3.py) — two-stage trainer. Stage A unconditional contrastive; stage B query alignment with graph encoder frozen.

## Training recipe (non-obvious bits)

1. **Boundary saturation mitigation carries over.** Small-gain Xavier on `node_in` (`gain=0.05`), learnable `tangent_scale` (init 0.1), radial-reg decay from `--radial-reg-weight` to `--radial-reg-weight-end` (floor > 0; never zero). See [../../CLAUDE.md](../../CLAUDE.md) known-issues section. Decay floor is non-negotiable for stage A.
2. **InfoNCE temperature defaults to 1.0**, sweep `{1.0, 3.0, 10.0}`. Hyperbolic distances span roughly `[0, 10]`; `τ<1` usually saturates softmax and indicates origin collapse (distances bunched near zero). If `τ=0.3` wins, investigate `|h|_mean` before accepting the number.
3. **k-hop neighbor exclusion default k=1.** Excludes self + 1-hop from the negative pool. Log `eff_negs_per_anchor` — the mask can silently eat the negative pool in dense regions; that's a training bug, not a model bug.
4. **Stage B must be ranking, not regression.** `pairwise_ranking_loss` with `margin=0.5` is the default. If it plateaus early, try `listwise_ranking_loss` via `--stage-b-loss listwise`. Do not reintroduce MSE.
5. **Query encoder is stage-B only.** Graph encoder is frozen before stage B starts (`requires_grad=False` on all non-query-encoder parameters). Stage-A is never aware of any task.

## Anti-patterns specific to v3

- **Don't concatenate the query into the graph encoder.** That's the v1/v2 pattern. v3's primary architectural claim is that the graph encoder is query-agnostic; wiring a query into it reverts to the v2 design.
- **Don't use `x[i]` cosine similarity as a positive-pair signal.** It's a tautology — the encoder already sees `x`. Use the edge signal and the same-label-*different*-features signal (low-cos constraint forces the abstraction).
- **Don't use inter-graph negatives** without a very good reason. Different graphs = different tasks = too-easy negatives; the model learns to distinguish "my graph" from "other graphs" instead of within-graph similarity.
- **Don't delete the radial-reg decay** even when contrastive loss looks healthy. Without it, InfoNCE pushes negatives outward and the model slowly boundary-saturates — the failure manifests only after many epochs.

## Success criteria (tightened from the plan)

- **Query-agnosticism sanity (primary):** Freeze v3 graph encoder after stage A; train `QueryToBall` on 25% of stage-B data (`--train-frac 0.25`); eval on full val. Success = nDCG@10 within 0.03 of `--train-frac 1.0`. This is the actual test of whether stage A produced a general embedding.
- **Inter-graph consistency (primary):** Train stage A on half the corpus, freeze, run encoder on held-out graphs. Silhouette on node-type labels ≥ 0.1 on held-out AND within 0.05 of trained-half silhouette.
- **Geometric claim:** v3_hyp mean nDCG@10 > v3_euc mean nDCG@10 by > one combined std on ≥ 4/6 tasks.
- **Downstream parity:** v3_hyp within 0.02 of v2 on ≥ 4/6 tasks, strictly better on ≥ 1/6 (with three-seed std tracked per task).
- **Intrinsic:** `nn_edge_precision@5` well above `random_baseline`; `silhouette > 0` on node-type labels; `nn_label_purity@5` above `random_baseline`.

## When to touch this folder

- Any work on embedding-first training or self-supervised graph pretraining.
- Ablations on positive-pair signals, temperature, k-hop exclusion, pairwise-vs-listwise stage-B.
- Do not port v3 patterns back into v1/v2 — they're separate research arms with different reproducibility baselines.
