r"""Build the deployable missing-link-suggester scorer artifact (v0, h32).

The h32 suggester experiment (runs/blend_h32_suggester) validated the
scoring recipe: val-gated routed blend<->mixture 0.100+-0.002 test|nonlocal
(blend-only 0.096; bfs 0.070). The experiment discards its nets; this
script persists a SERVING artifact so the Context Service endpoint
(Docs/V2_CHAIN_AND_SUGGESTER_PLAN.md) can load and score without training:

  runs/suggester_v0_h32_artifact/
    blend_nets.pt        per-task blend MLP state dicts (seed 0 recipe)
    mixture_offsets.pt   per-task K=3 gyro-frame tangent offsets
    manifest.json        trunk ckpt + encoder SHA, routing choices
                         (val-gated, do-no-harm default=blend), feature
                         spec, pinned reference scores for 3 held-out
                         cases (reload smoke asserts bit-equality)

    PYTHONIOENCODING=utf-8 py -m scripts.build_suggester_artifact
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.blend_pool_experiment import (
    _Ctx, _eval_arms, _fit_mixture, _route, _train_blend,
)
from src.codegraph import cases as C
from src.modelsv3.lock_baseline import sha256_file

CKPT = "runs/width-h32-hyp-l4-s0"
OUT = Path("runs/suggester_v0_h32_artifact")
SEED = 0
ARGS = SimpleNamespace(epochs=10, lr=1e-3, neg_sample=256, mixture_feats=3)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    ctx = _Ctx(Path("../tutorstructure_patch"), Path(CKPT), OUT, device)
    C.assign_file_split(ctx.cases, 0, (0.70, 0.15, 0.15))
    tasks = sorted({c.task for c in ctx.cases})
    train_cases = [c for c in ctx.cases if c.split == "train"]
    eval_cases = [c for c in ctx.cases if c.split in ("val", "test")]
    print(f"{len(train_cases)} train / {len(eval_cases)} eval; tasks {tasks}")

    V = _fit_mixture(ctx, train_cases, ARGS.mixture_feats, SEED)
    nets = {}
    for task in tasks:
        tr = [c for c in train_cases if c.task == task]
        if tr:
            print(f"  blend[{task}] n={len(tr)}")
            nets[task] = _train_blend(ctx, tr, ARGS, SEED)

    rows = _eval_arms(ctx, eval_cases, {}, nets, V_by_task=V,
                      k_mix=ARGS.mixture_feats)
    routed_test, choices = _route(rows, ("blend", "mixture"))
    print(f"routed test|nonlocal (seed {SEED}): {routed_test:.4f}; "
          f"choices: {choices}")

    torch.save({t: n.state_dict() for t, n in nets.items()},
               OUT / "blend_nets.pt")
    torch.save(V, OUT / "mixture_offsets.pt")

    # pinned reload-smoke references: first 3 eval cases with both arms
    pins = []
    for cs in eval_cases:
        if len(pins) >= 3 or cs.task not in nets or cs.task not in V:
            continue
        cand = sorted(set(
            ctx.pools.get(cs.task, torch.zeros(0).numpy()).tolist())
            | {r for r in cs.pos_rows if r != C.ABSTAIN_ROW})[:64]
        if len(cand) < 5:
            continue
        with torch.no_grad():
            sc = nets[cs.task](ctx.features(cs, cand)).squeeze(-1)
        pins.append({"case_id": cs.case_id, "task": cs.task,
                     "cand": [int(x) for x in cand],
                     "blend_scores_first5": [round(float(v), 6)
                                             for v in sc[:5]]})

    manifest = {
        "artifact": "missing-link-suggester scorer v0",
        "trunk_ckpt": CKPT,
        "encoder_sha256": sha256_file(Path(CKPT) / "encoder.pt"),
        "recipe": vars(ARGS) | {"seed": SEED},
        "routing_choices_val_gated": choices,
        "routing_default": "blend",
        "routed_test_nonlocal_ndcg10_seed0": routed_test,
        "reference_3seed": {"routed": "0.100+-0.002", "blend": "0.096+-0.001",
                            "bfs": 0.070,
                            "source": "runs/blend_h32_suggester"},
        "n_feats": ctx.n_feats,
        "feature_spec": "blend_pool_experiment._Ctx.features (base block; "
                        "mixture block only for basis=mixture via "
                        "mixture_offsets.mixture_score)",
        "pinned_cases": pins,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # reload smoke: nets round-trip + pinned scores reproduce
    nets2 = torch.load(OUT / "blend_nets.pt", map_location="cpu")
    from torch import nn
    for t, sd in nets2.items():
        m = nn.Sequential(nn.Linear(ctx.n_feats, 64), nn.ReLU(),
                          nn.Linear(64, 1))
        m.load_state_dict(sd)
        m.eval()
        nets2[t] = m
    case_by_id = {c.case_id: c for c in eval_cases}
    for p in pins:
        cs = case_by_id[p["case_id"]]
        with torch.no_grad():
            sc = nets2[p["task"]](
                ctx.features(cs, p["cand"])).squeeze(-1)
        got = [round(float(v), 6) for v in sc[:5]]
        assert got == p["blend_scores_first5"], (p["case_id"], got)
    print(f"reload smoke: {len(pins)} pinned cases bit-reproduce. "
          f"artifact: {OUT}")


if __name__ == "__main__":
    main()
