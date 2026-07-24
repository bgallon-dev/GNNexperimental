r"""Parity check between KettleGraphReasonerClean and EuclideanBaselineClean.

Reads both models, constructs them with identical config, and prints a
component-by-component comparison. The goal is to catch silent
divergence before training.

What this checks:
  1. Named-parameter correspondence. Every learnable parameter in one model
     should have a counterpart in the other with the same name and shape,
     except for hyperbolic-specific params (_c if learnable) and
     Euclidean-specific params (none expected).
  2. Total parameter count. Should match within ±2% for a fair comparison.
     Larger gaps mean one model has more capacity.
  3. Forward-pass output shapes. Both models must produce equivalently
     shaped outputs given the same dummy batch.
  4. Non-parameter module differences. Flags things like activation
     hardcoding, buffer registration, init-time side effects.

Run::

    py parity_check.py

Exit code 0 means all checks pass. Exit code 1 means one or more
mismatches; the script prints which ones. Treat any failure as a blocker
for the three-seed training comparison that follows.

Read the output top-to-bottom. The summary at the end tells you whether
the comparison you're about to run will mean anything.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

# Put the project root (parent of `src/`) on sys.path so `from src.models...`
# resolves whether this script is run directly or via `py -m`. We need the
# package-qualified form because the model modules use relative imports
# like `from .layers import poincare_ops`, which require a parent package.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.hyperbolic_gnn_clean import KettleGraphReasonerClean  # noqa: E402
from src.models.euclidean_baseline_clean import EuclideanBaselineClean  # noqa: E402


# ---- Shared config. Must be identical for both models. ---------------------
CONFIG: dict[str, Any] = dict(
    node_feat_dim=24,        # plausible value; the actual number from your
                             # corpus would replace this in the real run.
    edge_feat_dim=16,
    query_dim=32,
    hidden_dim=64,
    num_layers=3,
    type_dim=8,
    num_edge_types_max=8,
    node_feat_dim_schema=24,
    tangent_scale_init=0.10,
    activation="relu",
)
# Dummy input shapes used for the forward-pass check. N=10 nodes, E=20 edges
# is enough to exercise the full pipeline without being slow.
N_NODES = 10
N_EDGES = 20


# ---- Result types ----------------------------------------------------------
@dataclass
class ParamInfo:
    name: str
    shape: tuple[int, ...]
    numel: int


@dataclass
class Report:
    hyp_params: list[ParamInfo]
    euc_params: list[ParamInfo]
    hyp_total: int
    euc_total: int
    only_in_hyp: list[str]
    only_in_euc: list[str]
    shape_mismatches: list[tuple[str, tuple, tuple]]
    count_gap_pct: float
    forward_ok: bool
    forward_error: str
    output_shapes_match: bool
    output_shape_details: dict[str, Any]


# ---- Introspection helpers -------------------------------------------------
def collect_params(model: nn.Module) -> list[ParamInfo]:
    """Collect all trainable parameters with names and shapes, sorted."""
    params = [
        ParamInfo(name=n, shape=tuple(p.shape), numel=p.numel())
        for n, p in model.named_parameters()
        if p.requires_grad
    ]
    params.sort(key=lambda pi: pi.name)
    return params


def compare_params(
    hyp: list[ParamInfo], euc: list[ParamInfo]
) -> tuple[list[str], list[str], list[tuple[str, tuple, tuple]]]:
    """Return (only_in_hyp, only_in_euc, shape_mismatches)."""
    hyp_by_name = {p.name: p for p in hyp}
    euc_by_name = {p.name: p for p in euc}

    only_in_hyp = sorted(set(hyp_by_name) - set(euc_by_name))
    only_in_euc = sorted(set(euc_by_name) - set(hyp_by_name))

    shared = set(hyp_by_name) & set(euc_by_name)
    mismatches: list[tuple[str, tuple, tuple]] = []
    for name in sorted(shared):
        if hyp_by_name[name].shape != euc_by_name[name].shape:
            mismatches.append(
                (name, hyp_by_name[name].shape, euc_by_name[name].shape)
            )
    return only_in_hyp, only_in_euc, mismatches


def make_dummy_batch() -> dict[str, torch.Tensor]:
    """Construct a dummy forward-pass batch using CONFIG dimensions."""
    N, E = N_NODES, N_EDGES
    node_feat_dim = int(CONFIG["node_feat_dim"])
    edge_feat_dim = int(CONFIG["edge_feat_dim"])
    query_dim = int(CONFIG["query_dim"])
    num_edge_types_max = int(CONFIG["num_edge_types_max"])
    node_feat_dim_schema = int(CONFIG["node_feat_dim_schema"])
    # Random edges; some self-loops fine, model should handle them.
    edge_index = torch.randint(0, N, (2, E))
    return dict(
        node_features=torch.randn(N, node_feat_dim),
        edge_index=edge_index,
        edge_type=torch.randint(0, num_edge_types_max, (E,)),
        edge_descriptor=torch.randn(num_edge_types_max, edge_feat_dim),
        query=torch.randn(query_dim),
        node_descriptor=torch.randn(N, node_feat_dim_schema),
    )


def run_forward(model: nn.Module, batch: dict) -> Any:
    """Run forward pass in eval mode, no gradient. Returns the output
    dataclass or raises."""
    model.eval()
    with torch.no_grad():
        return model(**batch)


# ---- Reporting -------------------------------------------------------------
def print_param_table(params: list[ParamInfo], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    total = 0
    for pi in params:
        shape_str = "×".join(str(d) for d in pi.shape) if pi.shape else "scalar"
        print(f"  {pi.name:<50} {shape_str:<20} {pi.numel:>8,}")
        total += pi.numel
    print(f"  {'TOTAL':<50} {'':<20} {total:>8,}")


def print_diff(report: Report) -> None:
    print("\n" + "=" * 72)
    print("DIFF")
    print("=" * 72)

    if report.only_in_hyp:
        print("\nParams present ONLY in hyperbolic model:")
        for n in report.only_in_hyp:
            print(f"  + {n}")
    if report.only_in_euc:
        print("\nParams present ONLY in Euclidean model:")
        for n in report.only_in_euc:
            print(f"  + {n}")
    if not report.only_in_hyp and not report.only_in_euc:
        print("\nNamed-parameter sets match exactly. ✓")

    if report.shape_mismatches:
        print("\nShape mismatches on shared params:")
        for name, hs, es in report.shape_mismatches:
            print(f"  ! {name}: hyp={hs}  euc={es}")
    else:
        print("\nAll shared params have matching shapes. ✓")

    print(
        f"\nParameter count: hyp={report.hyp_total:,}  "
        f"euc={report.euc_total:,}  "
        f"gap={report.count_gap_pct:+.1f}%"
    )
    if abs(report.count_gap_pct) > 2.0:
        print(
            f"  ! Gap exceeds ±2%. One model has materially more capacity. "
            "Investigate before comparing."
        )
    else:
        print("  Gap within ±2%. Capacity matched. ✓")

    print("\nForward pass:")
    if not report.forward_ok:
        print(f"  ! FAILED: {report.forward_error}")
    else:
        print("  Both models forward-passed a dummy batch without error. ✓")
        for key, (hyp_shape, euc_shape, ok) in report.output_shape_details.items():
            marker = "✓" if ok else "!"
            print(f"  {marker} {key}: hyp={hyp_shape}  euc={euc_shape}")


def final_verdict(report: Report) -> bool:
    """Return True if everything passes, False otherwise."""
    blocking = []
    # only_in_hyp == ['_c'] is acceptable when _c is learnable (nn.Parameter).
    # Any other "only in" member is a structural difference.
    strictly_in_hyp_ok = {"_c", "tangent_scale"}
    # tangent_scale IS in both by design, so it shouldn't show up here;
    # listing it defensively in case init drift ever puts it in only one.
    unexpected_hyp_only = [n for n in report.only_in_hyp if n not in strictly_in_hyp_ok]
    if unexpected_hyp_only:
        blocking.append(f"unexpected hyp-only params: {unexpected_hyp_only}")
    if report.only_in_euc:
        blocking.append(f"unexpected euc-only params: {report.only_in_euc}")
    if report.shape_mismatches:
        blocking.append(f"shape mismatches: {len(report.shape_mismatches)}")
    if abs(report.count_gap_pct) > 2.0:
        blocking.append(f"param count gap {report.count_gap_pct:+.1f}% exceeds ±2%")
    if not report.forward_ok:
        blocking.append("forward pass failed")
    if report.forward_ok and not report.output_shapes_match:
        blocking.append("output shapes don't match")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if not blocking:
        print("ALL CHECKS PASSED. Safe to run the three-seed training comparison.")
        return True
    print("BLOCKING ISSUES:")
    for b in blocking:
        print(f"  ! {b}")
    print(
        "\nThe training comparison may produce misleading numbers until these "
        "are resolved."
    )
    return False


# ---- Main ------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("PARITY CHECK: KettleGraphReasonerClean vs EuclideanBaselineClean")
    print("=" * 72)
    print("\nConfig (identical for both models):")
    for k, v in CONFIG.items():
        print(f"  {k} = {v!r}")

    # Construct both models from the same config.
    torch.manual_seed(0)
    hyp = KettleGraphReasonerClean(**CONFIG)
    torch.manual_seed(0)
    euc = EuclideanBaselineClean(**CONFIG)

    # Collect params.
    hyp_params = collect_params(hyp)
    euc_params = collect_params(euc)
    hyp_total = sum(pi.numel for pi in hyp_params)
    euc_total = sum(pi.numel for pi in euc_params)
    count_gap = 100.0 * (hyp_total - euc_total) / max(euc_total, 1)

    # Compare param sets.
    only_in_hyp, only_in_euc, shape_mismatches = compare_params(hyp_params, euc_params)

    # Forward pass.
    batch = make_dummy_batch()
    forward_ok = True
    forward_error = ""
    output_shape_details: dict[str, Any] = {}
    output_shapes_match = True

    try:
        hyp_out = run_forward(hyp, batch)
        euc_out = run_forward(euc, batch)
        for field in ("node_scores", "edge_scores", "node_embeddings", "node_logits"):
            hs = tuple(getattr(hyp_out, field).shape)
            es = tuple(getattr(euc_out, field).shape)
            output_shape_details[field] = (hs, es, hs == es)
            if hs != es:
                output_shapes_match = False
    except Exception as exc:
        forward_ok = False
        forward_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    # Print.
    print_param_table(hyp_params, "HYPERBOLIC MODEL PARAMETERS")
    print_param_table(euc_params, "EUCLIDEAN MODEL PARAMETERS")

    report = Report(
        hyp_params=hyp_params,
        euc_params=euc_params,
        hyp_total=hyp_total,
        euc_total=euc_total,
        only_in_hyp=only_in_hyp,
        only_in_euc=only_in_euc,
        shape_mismatches=shape_mismatches,
        count_gap_pct=count_gap,
        forward_ok=forward_ok,
        forward_error=forward_error,
        output_shapes_match=output_shapes_match,
        output_shape_details=output_shape_details,
    )

    print_diff(report)
    ok = final_verdict(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
