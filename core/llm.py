"""
LLM wrapper using HuggingFace Inference API (free tier).
Primary: Qwen/Qwen2.5-72B-Instruct  (best reasoning, free on HF)
Fallback: mistralai/Mistral-7B-Instruct-v0.3
No local GPU required.
"""

import os
import json
import re
import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models"

# Model preference order (free on HuggingFace Inference API)
MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]


class LLMClient:
    """
    Calls HuggingFace Inference API (free).
    Set HF_TOKEN env variable with your HuggingFace read token.
    Get one free at https://huggingface.co/settings/tokens
    """

    def __init__(self, hf_token: Optional[str] = None):
        self.token = hf_token or os.environ.get("HF_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.current_model_idx = 0

    def _get_model(self) -> str:
        return MODELS[self.current_model_idx % len(MODELS)]

    def _try_next_model(self):
        self.current_model_idx += 1
        logger.warning(f"Switching to fallback model: {self._get_model()}")

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        retries: int = 3,
    ) -> str:
        """
        Make a chat-style call to HuggingFace Inference API.
        Returns the assistant response string.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "inputs": self._format_messages(messages),
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "return_full_text": False,
            },
        }

        for attempt in range(retries):
            model = self._get_model()
            url = f"{HF_API_URL}/{model}"
            try:
                resp = requests.post(url, headers=self.headers, json=payload, timeout=60)
                if resp.status_code == 503:
                    # Model loading, wait and retry
                    wait = resp.json().get("estimated_time", 20)
                    logger.info(f"Model loading, waiting {wait:.0f}s...")
                    time.sleep(min(wait, 30))
                    continue
                if resp.status_code == 429:
                    logger.info("Rate limited, waiting 10s...")
                    time.sleep(10)
                    continue
                if resp.status_code != 200:
                    logger.warning(f"HTTP {resp.status_code} from {model}, trying next model")
                    self._try_next_model()
                    continue

                result = resp.json()
                if isinstance(result, list) and result:
                    text = result[0].get("generated_text", "")
                    return self._clean_response(text)
                elif isinstance(result, dict):
                    text = result.get("generated_text", "")
                    return self._clean_response(text)

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt+1}")
            except Exception as e:
                logger.error(f"LLM call error: {e}")
                if attempt == retries - 1:
                    raise

        return "I encountered an error generating a response. Please try again."

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> dict:
        """
        Call LLM and parse response as JSON.
        Returns parsed dict or empty dict on failure.
        """
        system_prompt = system_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no backticks, no extra text."
        raw = self.call(system_prompt, user_prompt, max_tokens=max_tokens, temperature=0.3)
        return self._parse_json(raw)

    def _format_messages(self, messages: list) -> str:
        """Format chat messages into a single prompt string."""
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}<|end|>")
            elif role == "user":
                parts.append(f"<|user|>\n{content}<|end|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}<|end|>")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def _clean_response(self, text: str) -> str:
        """Strip any leftover prompt tokens from response."""
        # Remove common chat template artifacts
        for token in ["<|end|>", "<|assistant|>", "<|user|>", "<|system|>", "<|eot_id|>"]:
            text = text.replace(token, "")
        return text.strip()

    def _parse_json(self, text: str) -> dict:
        """Try to extract and parse JSON from LLM response."""
        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        # Try finding any JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        logger.warning(f"Failed to parse JSON from: {text[:200]}")
        return {}
