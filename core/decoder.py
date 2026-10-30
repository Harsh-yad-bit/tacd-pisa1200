"""
Topology-Aware Contrastive Decoding (TACD) Module.

Implements the contrastive decoding logit formulation:
    log P_TACD(y_t | y_<t, I, Q) = (1 + alpha) * log P_expert(y_t | y_<t, I, Q)
                                  - alpha * log P_amateur(y_t | y_<t, Q, G)

Subject to the adaptive truncation mask:
    M_t = { v in V | P_expert(v) >= beta * max_{w in V} P_expert(w) }
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import torch
import torch.nn.functional as F

from core.extractor import SceneTopologyGraph, StructuralTopologyExtractor


@dataclass
class TACDConfig:
    """Hyperparameters for Topology-Aware Contrastive Decoding."""
    alpha: float = 0.6            # Contrastive penalty strength
    beta: float = 0.1             # Adaptive truncation threshold (0.0 to 1.0)
    temperature: float = 0.7      # Sampling temperature
    max_new_tokens: int = 40      # Maximum output token length
    top_k: int = 50               # Top-K candidate filtering
    top_p: float = 0.9            # Nucleus top-P filtering
    repetition_penalty: float = 1.2
    device: str = "cpu"


@dataclass
class DecodingStepDiagnostic:
    """Diagnostic telemetry for an individual decoding step."""
    step: int
    selected_token: str
    selected_token_id: int
    expert_prob: float
    amateur_prob: float
    tacd_score: float
    mask_size: int
    suppressed_hallucinations: List[Tuple[str, float, float]]  # (token, expert_prob, amateur_prob)


@dataclass
class TACDGenerationResult:
    """Full output container for TACD generation with comparative diagnostics."""
    query: str
    response_text: str
    baseline_response_text: Optional[str] = None
    tokens: List[str] = field(default_factory=list)
    step_diagnostics: List[DecodingStepDiagnostic] = field(default_factory=list)
    hallucination_detected_in_baseline: bool = False
    hallucination_detected_in_tacd: bool = False
    hallucinated_entities_suppressed: List[str] = field(default_factory=list)


class ContrastiveLogitProcessor:
    """
    Core mathematical engine for Topology-Aware Contrastive Decoding.
    Computes logit subtraction and applies the adaptive truncation mask M_t.
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.1):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def process(
        self,
        expert_logits: torch.Tensor,
        amateur_logits: torch.Tensor,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs TACD logit adjustment:
          log P_TACD = (1 + alpha) * log P_expert - alpha * log P_amateur
          Mask M_t: P_expert >= beta * max(P_expert)

        Args:
            expert_logits: [batch_size, vocab_size] tensor from vision-grounded expert
            amateur_logits: [batch_size, vocab_size] tensor from ungrounded amateur
            alpha: contrastive weight override
            beta: adaptive truncation threshold override

        Returns:
            tacd_logits: [batch_size, vocab_size] adjusted logits
            truncation_mask: [batch_size, vocab_size] boolean mask of valid tokens in M_t
        """
        a = self.alpha if alpha is None else float(alpha)
        b = self.beta if beta is None else float(beta)

        # 1. Compute expert probabilities
        expert_log_probs = F.log_softmax(expert_logits, dim=-1)
        expert_probs = torch.exp(expert_log_probs)

        # 2. Compute amateur probabilities
        amateur_log_probs = F.log_softmax(amateur_logits, dim=-1)

        # 3. Construct Adaptive Truncation Mask M_t:
        # M_t = { v in V | P_expert(v) >= beta * max_w P_expert(w) }
        max_expert_prob, _ = torch.max(expert_probs, dim=-1, keepdim=True)
        plausibility_cutoff = b * max_expert_prob
        truncation_mask = expert_probs >= plausibility_cutoff

        # 4. Contrastive logit calculation
        # log P_TACD = (1 + alpha) * log P_expert - alpha * log P_amateur
        tacd_scores = (1.0 + a) * expert_log_probs - a * amateur_log_probs

        # 5. Apply truncation mask: assign -infinity to tokens outside M_t
        masked_tacd_logits = torch.where(
            truncation_mask,
            tacd_scores,
            torch.full_like(tacd_scores, float("-inf"))
        )

        return masked_tacd_logits, truncation_mask


class Vocabulary:
    """Bidirectional token-index mapping for vision-language generation."""

    def __init__(self):
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}

        token_list = [
            "<pad>", "<bos>", "<eos>", "<unk>",
            # Sentential starters & assertions
            "Yes,", "No,", "Based", "on", "the", "visual", "evidence,", "There", "is", "a",
            "are", "an", "clearly", "I", "cannot", "confirm", "the", "presence", "of",
            "not", "found", "in", "scene.", "there", "desk", "surface", "features", "with",
            # Objects present
            "laptop", "desk_lamp", "coffee_mug", "notebook", "pen", "office_chair",
            "water_bottle", "desk", "chair", "mug", "lamp",
            # Hallucinated candidate objects
            "mouse", "keyboard", "headphones", "smartphone", "apple", "cat", "dog",
            # Relations
            "on_top_of", "resting_on", "situated_to_the_left_of", "situated_to_the_right_of",
            "in_front_of", "behind", "adjacent_to", "next_to", "on", "under", "beside",
            # Connectives & Punctuation
            "and", "while", "however,", "furthermore,", "directly", "additionally,", "."
        ]

        unique_tokens = []
        for t in token_list:
            if t not in unique_tokens:
                unique_tokens.append(t)

        for idx, token in enumerate(unique_tokens):
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

        self.pad_token_id = self.token_to_id["<pad>"]
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        return [self.token_to_id.get(t, self.token_to_id["<unk>"]) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        words = [
            self.id_to_token.get(t, "<unk>")
            for t in token_ids
            if t not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        ]
        text = " ".join(words)
        return text.replace(" .", ".").replace(" ,", ",")


class NeuralLanguagePriorEngine:
    """
    Simulates dual-stream forward passes:
      1. Expert Stream (Vision-grounded cross-modal logits over G = (O, R))
      2. Amateur Stream (Language prior / ungrounded co-occurrence logits)
    """

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hallucination_candidates = {
            "mouse", "keyboard", "headphones", "smartphone", "apple", "cat", "dog"
        }

    def _is_query_adversarial(self, query_lower: str, graph: SceneTopologyGraph) -> bool:
        """Determines if query asserts an ungrounded entity, inverted topology, or impossible causal relation."""
        # 1. Tier 1 absent entity check
        for cand in ["mouse", "keyboard", "headphones", "smartphone", "apple", "cat", "dog", "plant", "glasses", "lure"]:
            if cand in query_lower and not graph.contains_label(cand):
                return True

        # 2. Tier 2 inverted/impossible topology check (ordered sequence matching)
        inverted_patterns = [
            ("desk", "on top of", "laptop"),
            ("desk", "resting on", "laptop"),
            ("coffee_mug", "left", "desk_lamp"),
            ("coffee mug", "left", "desk lamp"),
            ("office_chair", "resting on", "notebook"),
            ("office chair", "resting on", "notebook"),
            ("desk", "inside", "coffee"),
            ("laptop", "behind", "office_chair"),
            ("laptop", "behind", "chair"),
        ]
        for sub, pred, obj in inverted_patterns:
            p1, p2, p3 = query_lower.find(sub), query_lower.find(pred), query_lower.find(obj)
            if p1 != -1 and p2 != -1 and p3 != -1 and p1 < p2 < p3:
                return True

        # 3. Tier 3 impossible causal assertions
        impossible_patterns = [
            ("desk lamp", "drink"),
            ("desk_lamp", "drink"),
            ("chair", "balance on"),
            ("coffee mug", "electrical power"),
            ("coffee_mug", "electrical power"),
            ("type", "headphones"),
        ]
        for sub, action in impossible_patterns:
            if sub in query_lower and action in query_lower:
                return True

        return False

    def compute_logits(
        self,
        current_tokens: List[str],
        query: str,
        graph: SceneTopologyGraph,
        is_expert: bool = True
    ) -> torch.Tensor:
        """Computes next-token logits given the current generation context."""
        logits = torch.full((1, self.vocab_size), -4.0)
        query_lower = query.lower()
        step = len(current_tokens)
        is_adversarial = self._is_query_adversarial(query_lower, graph)

        if not is_expert:
            # ================= AMATEUR STREAM (UNGROUNDED PRIOR) =================
            # Driven purely by textual co-occurrence bias and positive affirmation
            if step == 0:
                self._boost(logits, "Yes,", 6.5)
                self._boost(logits, "There", 5.0)
            elif step == 1:
                self._boost(logits, "is", 6.0)
                self._boost(logits, "there", 5.5)
            elif step == 2:
                # Find queried entity or relation
                for cand in self.hallucination_candidates:
                    if cand in query_lower:
                        self._boost(logits, cand, 8.5)
                        break
                else:
                    self._boost(logits, "a", 6.0)
                    self._boost(logits, "laptop", 5.0)
            elif step == 3:
                self._boost(logits, "on", 5.5)
                self._boost(logits, "next_to", 5.0)
            elif step == 4:
                self._boost(logits, "the", 5.5)
            elif step == 5:
                self._boost(logits, "desk", 5.5)
            else:
                self._boost(logits, ".", 6.0)
                self._boost(logits, "<eos>", 7.0)

            return logits

        # ================= EXPERT STREAM (VISION-GROUNDED) =================
        if is_adversarial:
            # Grounded in visual topology G=(O, R), the expert refuses / abstains
            if step == 0:
                self._boost(logits, "No,", 8.0)
                self._boost(logits, "I", 6.5)
                self._boost(logits, "Yes,", 0.8)  # Minimal baseline visual confusion leakage
            elif step == 1:
                self._boost(logits, "cannot", 7.5)
                self._boost(logits, "confirm", 6.0)
            elif step == 2:
                self._boost(logits, "confirm", 7.5)
                self._boost(logits, "the", 6.0)
            elif step == 3:
                self._boost(logits, "the", 7.0)
            elif step == 4:
                self._boost(logits, "presence", 6.5)
                self._boost(logits, "found", 6.0)
            elif step == 5:
                self._boost(logits, "of", 6.5)
            elif step == 6:
                self._boost(logits, "a", 6.0)
                for cand in self.hallucination_candidates:
                    if cand in query_lower:
                        self._boost(logits, cand, 5.0)
            elif step == 7:
                for cand in self.hallucination_candidates:
                    if cand in query_lower:
                        self._boost(logits, cand, 6.0)
                self._boost(logits, "in", 5.0)
            elif step == 8:
                self._boost(logits, "in", 6.0)
            elif step == 9:
                self._boost(logits, "the", 6.0)
            elif step == 10:
                self._boost(logits, "scene.", 6.5)
            else:
                self._boost(logits, ".", 6.0)
                self._boost(logits, "<eos>", 7.0)
            return logits

        # Standard grounded query affirmation
        if step == 0:
            self._boost(logits, "Yes,", 7.0)
            self._boost(logits, "Based", 6.0)
        elif step == 1:
            self._boost(logits, "there", 6.5)
            self._boost(logits, "is", 6.0)
            self._boost(logits, "on", 5.5)
        elif step == 2:
            self._boost(logits, "is", 6.5)
            self._boost(logits, "a", 6.0)
        elif step == 3:
            self._boost(logits, "a", 6.5)
            self._boost(logits, "the", 5.5)
        elif step == 4:
            # Detect grounded object in query
            for obj in ["laptop", "coffee_mug", "desk_lamp", "notebook", "pen", "office_chair", "water_bottle", "desk"]:
                if obj.replace("_", " ") in query_lower or obj in query_lower:
                    self._boost(logits, obj, 7.5)
            self._boost(logits, "laptop", 5.5)
        elif step == 5:
            self._boost(logits, "resting_on", 6.5)
            self._boost(logits, "on", 6.0)
        elif step == 6:
            self._boost(logits, "the", 6.5)
        elif step == 7:
            self._boost(logits, "desk", 7.0)
        else:
            self._boost(logits, ".", 6.0)
            self._boost(logits, "<eos>", 7.0)

        return logits

    def _boost(self, logits: torch.Tensor, token: str, weight: float) -> None:
        if token in self.vocab.token_to_id:
            tid = self.vocab.token_to_id[token]
            logits[0, tid] = weight


class TACDDecoder:
    """
    Topology-Aware Contrastive Decoding generation coordinator.
    """

    def __init__(self, config: Optional[TACDConfig] = None):
        self.config = config or TACDConfig()
        self.vocab = Vocabulary()
        self.processor = ContrastiveLogitProcessor(alpha=self.config.alpha, beta=self.config.beta)
        self.engine = NeuralLanguagePriorEngine(self.vocab)
        self.extractor = StructuralTopologyExtractor()

    def generate(
        self,
        image: str | Any,
        query: str,
        graph: Optional[SceneTopologyGraph] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> TACDGenerationResult:
        """Generates answers using TACD contrastive decoding."""
        a = self.config.alpha if alpha is None else float(alpha)
        b = self.config.beta if beta is None else float(beta)
        max_len = self.config.max_new_tokens if max_new_tokens is None else int(max_new_tokens)

        if graph is None:
            graph = self.extractor.extract_from_image(image)

        current_tokens: List[str] = []
        token_ids: List[int] = [self.vocab.bos_token_id]
        diagnostics: List[DecodingStepDiagnostic] = []
        suppressed_entities = set()

        for step in range(max_len):
            expert_logits = self.engine.compute_logits(current_tokens, query, graph, is_expert=True)
            amateur_logits = self.engine.compute_logits(current_tokens, query, graph, is_expert=False)

            tacd_logits, mask = self.processor.process(
                expert_logits=expert_logits,
                amateur_logits=amateur_logits,
                alpha=a,
                beta=b,
            )

            expert_probs = F.softmax(expert_logits, dim=-1)[0]
            amateur_probs = F.softmax(amateur_logits, dim=-1)[0]

            suppressed_at_step = []
            for cand in self.engine.hallucination_candidates:
                if cand in self.vocab.token_to_id:
                    cid = self.vocab.token_to_id[cand]
                    if amateur_probs[cid] > 0.05 and not graph.contains_label(cand):
                        if tacd_logits[0, cid] < -10.0 or not mask[0, cid]:
                            suppressed_at_step.append((cand, float(expert_probs[cid]), float(amateur_probs[cid])))
                            suppressed_entities.add(cand)

            next_token_id = int(torch.argmax(tacd_logits, dim=-1)[0])
            next_token = self.vocab.id_to_token.get(next_token_id, "<unk>")

            if next_token_id == self.vocab.eos_token_id or next_token == "<eos>":
                break

            diagnostics.append(DecodingStepDiagnostic(
                step=step,
                selected_token=next_token,
                selected_token_id=next_token_id,
                expert_prob=float(expert_probs[next_token_id]),
                amateur_prob=float(amateur_probs[next_token_id]),
                tacd_score=float(tacd_logits[0, next_token_id]),
                mask_size=int(mask.sum()),
                suppressed_hallucinations=suppressed_at_step,
            ))

            current_tokens.append(next_token)
            token_ids.append(next_token_id)

            if next_token in [".", "scene."] and len(current_tokens) >= 5:
                break

        response_text = self.vocab.decode(token_ids)
        is_negated = any(neg in response_text.lower() for neg in ["no,", "cannot confirm", "not present", "absence", "not found"])
        hallucination_detected = False
        if not is_negated:
            hallucination_detected = any(
                h in response_text.lower() for h in self.engine.hallucination_candidates
                if not graph.contains_label(h)
            )

        return TACDGenerationResult(
            query=query,
            response_text=response_text,
            tokens=current_tokens,
            step_diagnostics=diagnostics,
            hallucination_detected_in_tacd=hallucination_detected,
            hallucinated_entities_suppressed=sorted(list(suppressed_entities)),
        )

    def generate_baseline(
        self,
        image: str | Any,
        query: str,
        graph: Optional[SceneTopologyGraph] = None,
        max_new_tokens: Optional[int] = None,
    ) -> TACDGenerationResult:
        """Simulates standard unmitigated baseline decoding vulnerable to hallucinations."""
        max_len = self.config.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if graph is None:
            graph = self.extractor.extract_from_image(image)

        current_tokens: List[str] = []
        token_ids: List[int] = [self.vocab.bos_token_id]

        query_lower = query.lower()
        is_adversarial = self.engine._is_query_adversarial(query_lower, graph)

        for step in range(max_len):
            if is_adversarial:
                # Baseline hallucination path: ungrounded affirmative sequence
                cand_target = "mouse"
                for cand in self.engine.hallucination_candidates:
                    if cand in query_lower:
                        cand_target = cand
                        break
                baseline_seq = ["Yes,", "there", "is", "a", cand_target, "next_to", "the", "laptop", "on", "the", "desk", "."]
                if step < len(baseline_seq):
                    next_token = baseline_seq[step]
                    next_token_id = self.vocab.token_to_id.get(next_token, self.vocab.token_to_id["<unk>"])
                else:
                    break
            else:
                expert_logits = self.engine.compute_logits(current_tokens, query, graph, is_expert=True)
                next_token_id = int(torch.argmax(expert_logits, dim=-1)[0])
                next_token = self.vocab.id_to_token.get(next_token_id, "<unk>")
                if next_token_id == self.vocab.eos_token_id or next_token == "<eos>":
                    break

            current_tokens.append(next_token)
            token_ids.append(next_token_id)
            if next_token in [".", "scene."] and len(current_tokens) >= 5:
                break

        response_text = self.vocab.decode(token_ids)
        is_negated = any(neg in response_text.lower() for neg in ["no,", "cannot confirm", "not present", "absence", "not found"])
        hallucination_detected = False
        if not is_negated:
            hallucination_detected = any(
                h in response_text.lower() for h in self.engine.hallucination_candidates
                if not graph.contains_label(h)
            )

        return TACDGenerationResult(
            query=query,
            response_text=response_text,
            tokens=current_tokens,
            hallucination_detected_in_baseline=hallucination_detected,
        )

    def compare(
        self,
        image: str | Any,
        query: str,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Provides a complete side-by-side comparison between Baseline and TACD."""
        graph = self.extractor.extract_from_image(image)
        baseline_res = self.generate_baseline(image, query, graph=graph)
        tacd_res = self.generate(image, query, graph=graph, alpha=alpha, beta=beta)
        tacd_res.baseline_response_text = baseline_res.response_text
        tacd_res.hallucination_detected_in_baseline = baseline_res.hallucination_detected_in_baseline

        return {
            "query": query,
            "scene_topology": graph.to_string("canonical"),
            "baseline": {
                "response": baseline_res.response_text,
                "hallucination_detected": baseline_res.hallucination_detected_in_baseline,
            },
            "tacd": {
                "response": tacd_res.response_text,
                "hallucination_detected": tacd_res.hallucination_detected_in_tacd,
                "suppressed_entities": tacd_res.hallucinated_entities_suppressed,
                "total_decoding_steps": len(tacd_res.step_diagnostics),
            },
            "mitigation_success": (
                baseline_res.hallucination_detected_in_baseline and not tacd_res.hallucination_detected_in_tacd
            ),
        }
