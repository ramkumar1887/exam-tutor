"""
LLM Client Wrapper backed by Load-Balanced, Fault-Tolerant LLMRouter.
Features:
- Per-model timeout (6s) & exponential backoff
- Dynamic routing strategies: Round-Robin, Least-Failures, Priority
- Per-model 3-state Circuit Breaker (CLOSED -> OPEN -> HALF-OPEN)
- In-memory telemetry and observability
- Zero local GPU required
"""

import os
import logging
from typing import Optional, Dict, Any, List

from core.router import LLMRouter, RoutingStrategy, CircuitState, ModelEndpoint, DEFAULT_MODELS

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Production LLM Client backed by LLMRouter.
    Automatically manages multi-model load balancing and circuit breaking.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN,
        models: Optional[List[str]] = None,
        timeout_s: float = 6.0,
        max_retries: int = 2,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_s: float = 20.0,
    ):
        self.token = hf_token or os.environ.get("HF_TOKEN", "")
        self.router = LLMRouter(
            hf_token=self.token,
            models=models or DEFAULT_MODELS,
            strategy=strategy,
            timeout_s=timeout_s,
            max_retries=max_retries,
            circuit_failure_threshold=circuit_failure_threshold,
            circuit_cooldown_s=circuit_cooldown_s,
        )

    def _get_model(self) -> str:
        """Returns primary active model name."""
        return self.router.model_names[0]

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        retries: int = 3,
    ) -> str:
        """Dispatches request through load-balanced router with circuit breaking & failover."""
        return self.router.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> dict:
        """Dispatches structured JSON generation through load-balanced router."""
        return self.router.call_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns per-endpoint and router-level telemetry metrics."""
        return self.router.get_telemetry()

    def simulate_model_failure(self, model_name: str, enable_failure: bool = True):
        """Simulates endpoint outage for resilience testing."""
        self.router.simulate_endpoint_failure(model_name, enable_failure)

    def reset_circuit_breakers(self):
        """Resets all endpoint circuit breakers."""
        self.router.reset_all_circuits()
