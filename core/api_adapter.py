"""
API Adapter for External Multimodal LLMs (Google Gemini & OpenAI ChatGPT).

Allows TACD to orchestrate live contrastive verification using:
  1. Google Gemini API (e.g. gemini-1.5-flash, gemini-2.0-flash)
  2. OpenAI ChatGPT API (e.g. gpt-4o, gpt-4o-mini)
"""

from __future__ import annotations
import os
import json
import base64
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, Tuple

from core.extractor import SceneTopologyGraph


def encode_image_base64(image_path: str) -> Tuple[str, str]:
    """Encodes a local image to base64 and detects MIME type."""
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded, mime


class ExternalVLMClient:
    """Dispatches multimodal requests to external Gemini or OpenAI APIs."""

    def __init__(self, provider: str = "gemini", model_name: Optional[str] = None):
        self.provider = provider.lower()
        if self.provider == "gemini":
            self.api_key = os.environ.get("GEMINI_API_KEY", "")
            self.model_name = model_name or "gemini-1.5-flash"
        elif self.provider in ["openai", "chatgpt"]:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
            self.model_name = model_name or "gpt-4o-mini"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def call_gemini(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Invokes Google Gemini API using pure standard library urllib."""
        if not self.api_key:
            return "[Error: GEMINI_API_KEY environment variable is not set. Export your key with: export GEMINI_API_KEY='your-key']"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        parts = []

        if image_path and os.path.exists(image_path):
            img_b64, mime = encode_image_base64(image_path)
            parts.append({
                "inline_data": {
                    "mime_type": mime,
                    "data": img_b64
                }
            })

        parts.append({"text": prompt})
        payload = json.dumps({"contents": [{"parts": parts}]}).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
                return "No response generated from Gemini."
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            return f"[Gemini API HTTP Error {e.code}: {err_body}]"
        except Exception as e:
            return f"[Gemini API Error: {str(e)}]"

    def call_openai(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Invokes OpenAI ChatGPT API using pure standard library urllib."""
        if not self.api_key:
            return "[Error: OPENAI_API_KEY environment variable is not set. Export your key with: export OPENAI_API_KEY='your-key']"

        url = "https://api.openai.com/v1/chat/completions"
        content_items = []

        if image_path and os.path.exists(image_path):
            img_b64, mime = encode_image_base64(image_path)
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"}
            })

        content_items.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content_items}]

        payload = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 150
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return "No response generated from OpenAI."
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            return f"[OpenAI API HTTP Error {e.code}: {err_body}]"
        except Exception as e:
            return f"[OpenAI API Error: {str(e)}]"

    def execute_tacd_contrastive_pass(
        self,
        image_path: str,
        query: str,
        graph: SceneTopologyGraph
    ) -> Dict[str, Any]:
        """
        Executes dual-stream contrastive verification using the external model:
          1. Baseline Expert Pass: Multimodal prompt with raw image.
          2. Topology-Aware Contrastive Pass: Prompts with extracted topology G = (O, R)
             as an explicit grounding prior constraint.
        """
        # Baseline prompt (prone to ungrounded co-occurrence hallucination)
        base_prompt = f"Answer the following question about this image concisely and directly: {query}"
        
        # TACD Topology Grounded prompt
        topology_summary = graph.to_string("canonical")
        tacd_prompt = (
            f"You are a strictly grounded visual reasoning system operating under TACD protocol.\n"
            f"Verified Topological Scene Graph G = (O, R):\n{topology_summary}\n\n"
            f"Instruction: You must verify any entity or spatial relationship against the visual evidence "
            f"and the topology graph above. If an object or relationship is absent or unverified, you MUST explicitly abstain "
            f"or state that it cannot be confirmed, rather than relying on language prior assumptions.\n\n"
            f"Question: {query}"
        )

        if self.provider == "gemini":
            base_resp = self.call_gemini(base_prompt, image_path=image_path)
            tacd_resp = self.call_gemini(tacd_prompt, image_path=image_path)
        else:
            base_resp = self.call_openai(base_prompt, image_path=image_path)
            tacd_resp = self.call_openai(tacd_prompt, image_path=image_path)

        return {
            "provider": self.provider,
            "model": self.model_name,
            "query": query,
            "baseline_response": base_resp,
            "tacd_response": tacd_resp,
        }
