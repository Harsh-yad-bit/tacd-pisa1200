"""
Unit and Integration Test Suite for TACD & PISA-1200.
"""

import math
import pytest
import torch
import torch.nn.functional as F

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
    Vocabulary,
)
from benchmarks.pisa1200_eval import (
    PISA1200Benchmark,
    run_pisa1200_evaluation,
)


def test_bounding_box_geometry():
    """Verify bounding box properties, containment, and IoU calculus."""
    b1 = BoundingBox(ymin=0.2, xmin=0.2, ymax=0.6, xmax=0.6)
    assert math.isclose(b1.width, 0.4)
    assert math.isclose(b1.height, 0.4)
    assert math.isclose(b1.area, 0.16)
    assert b1.center == (0.4, 0.4)

    # Identical box IoU must be 1.0
    assert math.isclose(b1.iou(b1), 1.0)

    # Completely disjoint box
    b2 = BoundingBox(ymin=0.7, xmin=0.7, ymax=0.9, xmax=0.9)
    assert b1.iou(b2) == 0.0

    # Contained box
    container = BoundingBox(ymin=0.1, xmin=0.1, ymax=0.9, xmax=0.9)
    assert math.isclose(b1.containment_ratio(container), 1.0)


def test_structural_topology_extractor():
    """Verify graph generation, entities, and spatial relations from sample image."""
    extractor = StructuralTopologyExtractor()
    graph = extractor.extract_from_image("assets/sample.jpg")

    assert len(graph.objects) >= 5
    assert graph.contains_label("laptop")
    assert graph.contains_label("desk")
    assert graph.contains_label("coffee_mug")

    # Check that directed relations are present
    verified, conf = graph.verify_relation("laptop", "on_top_of", "desk")
    assert verified
    assert conf > 0.8


def test_contrastive_logit_processor_math():
    """
    Verify exact mathematical formulation:
      log P_TACD = (1 + alpha) * log P_expert - alpha * log P_amateur
      Mask Mt = { v | P_expert(v) >= beta * max P_expert }
    """
    vocab_size = 5
    alpha = 0.5
    beta = 0.2

    processor = ContrastiveLogitProcessor(alpha=alpha, beta=beta)

    # Create synthetic logits
    # Token 0 is high in expert and amateur
    # Token 1 is low in expert, high in amateur (ungrounded hallucination)
    # Token 2 is very low in expert (should be masked out by beta)
    expert_logits = torch.tensor([[10.0, 5.0, 1.0, 8.0, 3.0]])
    amateur_logits = torch.tensor([[8.0, 9.0, 2.0, 5.0, 3.0]])

    tacd_logits, mask = processor.process(expert_logits, amateur_logits, alpha=alpha, beta=beta)

    # Verify expert probabilities and truncation cutoff
    expert_log_p = F.log_softmax(expert_logits, dim=-1)
    expert_p = torch.exp(expert_log_p)
    max_p = torch.max(expert_p).item()

    # Check mask validity: tokens with P_expert < beta * max_p must be masked with -inf
    for i in range(vocab_size):
        if expert_p[0, i].item() < beta * max_p:
            assert not mask[0, i].item()
            assert tacd_logits[0, i].item() == float("-inf")
        else:
            assert mask[0, i].item()
            # Check logit value calculation
            amateur_log_p = F.log_softmax(amateur_logits, dim=-1)
            expected_val = (1.0 + alpha) * expert_log_p[0, i] - alpha * amateur_log_p[0, i]
            assert math.isclose(tacd_logits[0, i].item(), expected_val.item(), rel_tol=1e-4)


def test_hallucination_suppression_e2e():
    """Verify that TACD suppresses an ungrounded lure query where baseline hallucinates."""
    decoder = TACDDecoder(config=TACDConfig(alpha=0.6, beta=0.1))
    comparison = decoder.compare(
        image="assets/sample.jpg",
        query="Is there a computer mouse next to the laptop on the desk?"
    )

    # Baseline should hallucinate affirmative presence of mouse
    assert comparison["baseline"]["hallucination_detected"]

    # TACD should refuse/abstain and flag no hallucination
    assert not comparison["tacd"]["hallucination_detected"]
    assert "mouse" in comparison["tacd"]["suppressed_entities"]
    assert comparison["mitigation_success"]


def test_pisa1200_benchmark_composition():
    """Verify PISA-1200 dataset composition: exactly 1200 items, 3 tiers of 400, 20% adversarial."""
    benchmark = PISA1200Benchmark(seed=42)
    assert len(benchmark.items) == 1200

    tier1 = [it for it in benchmark.items if it.tier == "tier1_identification"]
    tier2 = [it for it in benchmark.items if it.tier == "tier2_topology"]
    tier3 = [it for it in benchmark.items if it.tier == "tier3_causal"]

    assert len(tier1) == 400
    assert len(tier2) == 400
    assert len(tier3) == 400

    adv_items = [it for it in benchmark.items if it.is_adversarial]
    assert len(adv_items) == 240  # Exactly 20.0% of 1200
