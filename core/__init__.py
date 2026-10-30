"""
Topology-Aware Contrastive Decoding (TACD) Core Module.

Contains structural layout graph extraction and contrastive logit adjustment
mechanisms for mitigating cross-modal hallucinations in Vision-Language Models.
"""

from core.extractor import (
    BoundingBox,
    DetectedObject,
    SpatialRelation,
    SceneTopologyGraph,
    StructuralTopologyExtractor,
)
from core.decoder import (
    TACDConfig,
    ContrastiveLogitProcessor,
    TACDDecoder,
)

__all__ = [
    "BoundingBox",
    "DetectedObject",
    "SpatialRelation",
    "SceneTopologyGraph",
    "StructuralTopologyExtractor",
    "TACDConfig",
    "ContrastiveLogitProcessor",
    "TACDDecoder",
]
