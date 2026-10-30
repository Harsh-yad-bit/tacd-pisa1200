#!/usr/bin/env python3
"""
TACD & PISA-1200: Beyond Pixel Grounding CLI.

Official reference implementation for:
"Beyond Pixel Grounding: Mitigating Cross-Modal Hallucination in Vision-Language Models
via Contrastive Decoding and Structural Topology Priors (PISA-1200 & TACD)"
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Optional

from core.extractor import StructuralTopologyExtractor, SceneTopologyGraph
from core.decoder import TACDDecoder, TACDConfig
from core.api_adapter import ExternalVLMClient


def format_header() -> str:
    return (
        "\n" + "=" * 84 + "\n"
        "  TACD: Topology-Aware Contrastive Decoding for Vision-Language Models\n"
        "  Paper: Beyond Pixel Grounding: Mitigating Cross-Modal Hallucination via\n"
        "         Contrastive Decoding and Structural Topology Priors (PISA-1200 & TACD)\n"
        + "=" * 84
    )


def run_comparison(
    image_path: str,
    query: str,
    alpha: float = 0.6,
    beta: float = 0.1,
    output_path: Optional[str] = None,
    provider: str = "local",
    model_name: Optional[str] = None
) -> None:
    """Executes side-by-side contrastive comparison between Baseline and TACD."""
    print(format_header())
    print(f"\n[1/3] Loading Input Image & Query...")
    print(f"  • Image: {image_path}")
    print(f"  • Query: \"{query}\"")
    print(f"  • Hyperparameters: alpha = {alpha} (penalty), beta = {beta} (truncation)")

    if not os.path.exists(image_path):
        print(f"\n[ERROR] Image not found at path: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[2/3] Extracting Structural Layout Graph G = (O, R)...")
    extractor = StructuralTopologyExtractor()
    graph = extractor.extract_from_image(image_path)

    print(f"  ✓ Identified {len(graph.objects)} Grounded Entities:")
    for obj_id, obj in graph.objects.items():
        cx, cy = obj.bbox.center
        print(f"    - {obj.label:<14} (ID: {obj_id}) [bbox: ymin={obj.bbox.ymin:.2f}, xmin={obj.bbox.xmin:.2f}, ymax={obj.bbox.ymax:.2f}, xmax={obj.bbox.xmax:.2f}] (conf: {obj.confidence:.2f})")

    print(f"\n  ✓ Inferred {len(graph.relations)} Directed Topological Relations:")
    for r in graph.relations[:8]:
        sub = graph.objects[r.subject_id].label if r.subject_id in graph.objects else r.subject_id
        obj = graph.objects[r.object_id].label if r.object_id in graph.objects else r.object_id
        print(f"    - <{sub} --[{r.predicate}]--> {obj}> (conf: {r.confidence:.2f})")
    if len(graph.relations) > 8:
        print(f"    ... [{len(graph.relations) - 8} additional topological relations computed]")

    print(f"\n[3/3] Executing Decoding Passes (Baseline vs TACD)...")
    decoder = TACDDecoder(config=TACDConfig(alpha=alpha, beta=beta))
    comparison = decoder.compare(image_path, query, alpha=alpha, beta=beta)

    base_resp = comparison["baseline"]["response"]
    base_halluc = comparison["baseline"]["hallucination_detected"]
    tacd_resp = comparison["tacd"]["response"]
    tacd_halluc = comparison["tacd"]["hallucination_detected"]
    suppressed = comparison["tacd"]["suppressed_entities"]
    mitigated = comparison["mitigation_success"]

    print("\n" + "-" * 84)
    print("  DECODING COMPARISON RESULTS")
    print("-" * 84)
    print(f"  [Standard VLM Baseline (alpha = 0.0)]")
    print(f"  Response:        \"{base_resp}\"")
    print(f"  Hallucination:   {'[DETECTED - UNGROUNDED PRIOR]' if base_halluc else '[NOT DETECTED]'}")
    print()
    print(f"  [TACD Mitigated (alpha = {alpha}, beta = {beta})]")
    print(f"  Response:        \"{tacd_resp}\"")
    print(f"  Hallucination:   {'[DETECTED]' if tacd_halluc else '[CLEAN / GROUNDED]'}")
    print(f"  Suppressed Lures: {suppressed if suppressed else 'None'}")
    print(f"  Total Steps:     {comparison['tacd']['total_decoding_steps']}")
    print("-" * 84)

    if mitigated:
        print("  >>> OUTCOME: SUCCESSFUL HALLUCINATION SUPPRESSION")
        print("      Ungrounded prior penalty + adaptive truncation mask (Mt) successfully")
        print("      filtered unverified entity candidates while maintaining generation fluency.")
    elif not base_halluc and not tacd_halluc:
        print("  >>> OUTCOME: VERIFIED GROUNDED GENERATION")
        print("      Both models aligned with visual topology G = (O, R).")
    else:
        print("  >>> OUTCOME: COMPLETED WITH TELEMETRY")
    print("=" * 84 + "\n")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="TACD: Topology-Aware Contrastive Decoding CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default="assets/sample.jpg",
        help="Path to input image file (JPEG/PNG)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Is there a computer mouse next to the laptop on the desk?",
        help="Input query or prompt to evaluate",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Contrastive penalty strength alpha (higher = stronger suppression of language prior)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Adaptive truncation threshold beta (filters tokens with P_expert < beta * max(P_expert))",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "gemini", "openai"],
        default="local",
        help="Multimodal model provider: 'local' (built-in TACD engine), 'gemini' (Google Gemini API), or 'openai' (ChatGPT)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name for external provider (e.g. 'gemini-1.5-flash', 'gpt-4o')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["compare", "extract", "benchmark"],
        default="compare",
        help="Execution mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to export JSON output results",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1200,
        help="Number of samples to run when mode is 'benchmark'",
    )

    args = parser.parse_args()

    # Resolve relative paths
    if not os.path.isabs(args.image) and not os.path.exists(args.image):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(base_dir, args.image)
        if os.path.exists(candidate):
            args.image = candidate

    if args.mode == "compare":
        run_comparison(
            image_path=args.image,
            query=args.query,
            alpha=args.alpha,
            beta=args.beta,
            output_path=args.output,
            provider=args.provider,
            model_name=args.model,
        )
    elif args.mode == "extract":
        print(f"Extracting layout topology graph from: {args.image}")
        extractor = StructuralTopologyExtractor()
        graph = extractor.extract_from_image(args.image)
        print(graph.to_string("canonical"))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(graph.to_dict(), f, indent=2)
            print(f"Graph JSON saved to: {args.output}")
    elif args.mode == "benchmark":
        from benchmarks.pisa1200_eval import run_pisa1200_evaluation, print_evaluation_summary
        bench_out = args.output or "benchmarks/results/pisa1200_metrics.json"
        metrics = run_pisa1200_evaluation(
            image_path=args.image,
            num_samples=args.samples,
            alpha=args.alpha,
            beta=args.beta,
            output_path=bench_out,
        )
        print_evaluation_summary(metrics)


if __name__ == "__main__":
    main()
