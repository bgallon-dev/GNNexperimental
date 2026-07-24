r"""Evidence Coverage Reasoner (ECR) -- answer-completeness layer (T7).

Plan of record: ``Docs/EVIDENCE_COVERAGE_REASONER_PLAN.md``.

**The kernel now lives outside this repo**, in the standalone, dependency-free
``coverage_reasoner`` package (``../coverage-reasoner/``, installed with
``pip install -e ../coverage-reasoner``). It was extracted 2026-07-16: the
contracts, the deterministic slot/gate/verdict evaluator, and the seven frozen
answer-shape descriptors have no KGR coupling at all, and are useful to any
project asking "does this evidence support this claim?".

It was **moved, not copied**. A second implementation of a kernel whose whole
value is deterministic, reproducible verdicts would drift, and drift here means
two answers to the same question. This module re-exports it unchanged, so
``from src.evidence.coverage import evaluate, get_shape`` keeps working.

What stays here is ``compile.py`` -- the KGR *binding*, and the only part that
ever touched KGR types (``ResearchQuestion``, ``CandidateBundle``). It turns
this repo's portable artifacts into a canonical ``coverage.json``.

The boundary the kernel guarantees (no database, no LLM, no semantic
extraction, no confidence in the verdict) is unchanged and still asserted by
``tests/test_evidence_coverage.py``, which additionally pins this repo's
canonical serialization equal to the package's own copy -- if those two ever
disagree, identical artifacts hash differently.
"""

from __future__ import annotations

from coverage_reasoner.contracts import (
    COVERAGE_SCHEMA_VERSION,
    AnswerabilityVerdict,
    AnswerShape,
    AnswerSlot,
    ClassConstraint,
    CoverageArtifact,
    CoverageAssignment,
    CoverageContractError,
    CoverageManifest,
    EvidenceUniverse,
    FrontierTarget,
    HardGate,
    HardGateResult,
    NormalizationRef,
    ShapeSelection,
    SlotCoverage,
    SourceCoverage,
    SubClaim,
    VerdictReason,
)
from coverage_reasoner.evaluator import evaluate
from coverage_reasoner.shapes import (
    SHAPES_BY_ID,
    SHAPES_BY_ID_VERSION,
    STRESS_SHAPE_IDS,
    get_shape,
)

from .compile import (
    COVERAGE_COMPILER_VERSION,
    compile_coverage,
    coverage_bytes,
    valid_evidence_ids,
)

__all__ = [
    "COVERAGE_SCHEMA_VERSION",
    "COVERAGE_COMPILER_VERSION",
    # contracts (re-exported from the standalone kernel)
    "AnswerShape",
    "AnswerSlot",
    "ClassConstraint",
    "SubClaim",
    "HardGate",
    "ShapeSelection",
    "SourceCoverage",
    "EvidenceUniverse",
    "NormalizationRef",
    "CoverageAssignment",
    "SlotCoverage",
    "HardGateResult",
    "VerdictReason",
    "AnswerabilityVerdict",
    "FrontierTarget",
    "CoverageManifest",
    "CoverageArtifact",
    "CoverageContractError",
    # kernel + KGR binding
    "evaluate",
    "compile_coverage",
    "coverage_bytes",
    "valid_evidence_ids",
    # shape registry
    "SHAPES_BY_ID",
    "SHAPES_BY_ID_VERSION",
    "STRESS_SHAPE_IDS",
    "get_shape",
]
