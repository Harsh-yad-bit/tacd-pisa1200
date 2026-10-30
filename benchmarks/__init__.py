"""
PISA-1200 Benchmark Package.

Perceptual-Structural Inference and Spatial Alignment Benchmark (1200 items),
structured into Tier 1 (Entity Grounding), Tier 2 (Spatial Topology),
and Tier 3 (Causal Context), with a 20% unanswerable adversarial split.
"""

from benchmarks.pisa1200_eval import (
    PISA1200Benchmark,
    PISAItem,
    PISAEvalMetrics,
    run_pisa1200_evaluation,
)

__all__ = [
    "PISA1200Benchmark",
    "PISAItem",
    "PISAEvalMetrics",
    "run_pisa1200_evaluation",
]
