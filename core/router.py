"""
Load-Balanced, Fault-Tolerant LLM Router
Provides:
- Per-model endpoint abstraction with dedicated timeouts & exponential backoff retries
- Dynamic routing strategies: Round-Robin, Least-Failures / Health-Weighted, Priority-Failover
- Per-model 3-state Circuit Breaker (CLOSED -> OPEN -> HALF-OPEN)
- In-memory telemetry: request counts, success/failure rates, latency tracking, failover counters
- Deterministic mock support & simulated outage injection for resilience testing
"""

import os
import re
import time
import json
import logging
import threading
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models"

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]


class CircuitState(str, Enum):
    CLOSED = "CLOSED"         # Normal operation: traffic routed to model
    OPEN = "OPEN"             # Tripped: model skipped immediately without network wait
    HALF_OPEN = "HALF_OPEN"   # Probing: testing single request after cooldown window


class RoutingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_FAILURES = "least_failures"
    LEAST_LATENCY = "least_latency"
    PRIORITY_FAILOVER = "priority_failover"


class CircuitBreaker:
    """
    Thread-safe 3-state circuit breaker per model endpoint.
    """

    def __init__(
        self,
        model_name: str,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        half_open_success_threshold: int = 1,
    ):
        self.model_name = model_name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_success_threshold = half_open_success_threshold

        self.state: CircuitState = CircuitState.CLOSED
        self.consecutive_failures: int = 0
        self.consecutive_successes: int = 0
        self.last_failure_time: float = 0.0
        self.last_state_change_time: float = time.time()
        self.total_trips: int = 0
        self._lock = threading.Lock()

    def can_attempt(self) -> bool:
        """Determines if a request can be dispatched to this endpoint."""
        with self._lock:
            now = time.time()
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if (now - self.last_failure_time) >= self.cooldown_seconds:
                    logger.info(
                        f"[CircuitBreaker:{self.model_name}] Cooldown {self.cooldown_seconds}s elapsed. "
                        f"Transitioning OPEN -> HALF_OPEN (probing recovery)."
                    )
                    self.state = CircuitState.HALF_OPEN
                    self.consecutive_successes = 0
                    self.last_state_change_time = now
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
            return False

    def record_success(self):
        """Records successful invocation, recovering circuit if in HALF_OPEN."""
        with self._lock:
            now = time.time()
            self.consecutive_failures = 0
            if self.state == CircuitState.HALF_OPEN:
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.half_open_success_threshold:
                    logger.info(
                        f"[CircuitBreaker:{self.model_name}] Probe succeeded! "
                        f"Transitioning HALF_OPEN -> CLOSED (recovered)."
                    )
                    self.state = CircuitState.CLOSED
                    self.last_state_change_time = now
            elif self.state == CircuitState.OPEN:
                self.state = CircuitState.CLOSED
                self.last_state_change_time = now

    def record_failure(self, error_msg: str = ""):
        """Records failed invocation, tripping circuit to OPEN if threshold reached."""
        with self._lock:
            now = time.time()
            self.consecutive_failures += 1
            self.last_failure_time = now

            if self.state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"[CircuitBreaker:{self.model_name}] Probe failed in HALF_OPEN: {error_msg}. "
                    f"Tripping back to OPEN for {self.cooldown_seconds}s."
                )
                self.state = CircuitState.OPEN
                self.total_trips += 1
                self.last_state_change_time = now
            elif self.state == CircuitState.CLOSED:
                if self.consecutive_failures >= self.failure_threshold:
                    logger.warning(
                        f"[CircuitBreaker:{self.model_name}] {self.consecutive_failures} consecutive failures. "
                        f"Tripping circuit CLOSED -> OPEN for {self.cooldown_seconds}s. Reason: {error_msg}"
                    )
                    self.state = CircuitState.OPEN
                    self.total_trips += 1
                    self.last_state_change_time = now

    def reset(self):
        """Manually reset circuit breaker to healthy CLOSED state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.last_state_change_time = time.time()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state.value,
                "consecutive_failures": self.consecutive_failures,
                "consecutive_successes": self.consecutive_successes,
                "total_trips": self.total_trips,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_time": self.last_failure_time,
                "last_state_change_time": self.last_state_change_time,
            }


class ModelEndpoint:
    """
    Encapsulates a single model endpoint with dedicated timeout,
    exponential backoff retry policy, circuit breaker, and in-memory telemetry.
    """

    def __init__(
        self,
        model_name: str,
        hf_token: str = "",
        timeout_s: float = 6.0,
        max_retries: int = 2,
        backoff_base_s: float = 0.5,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_s: float = 20.0,
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s

        self.circuit_breaker = CircuitBreaker(
            model_name=model_name,
            failure_threshold=circuit_failure_threshold,
            cooldown_seconds=circuit_cooldown_s,
        )

        # In-memory Telemetry
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.recent_latencies: List[float] = []
        self.last_error: str = ""
        self.last_error_time: float = 0.0
        self._mock_failure: bool = False
        self._lock = threading.Lock()

    def execute_request(
        self,
        formatted_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Executes request with retries and exponential backoff against this model.
        Updates latency and health metrics.
        """
        if not self.circuit_breaker.can_attempt():
            raise RuntimeError(
                f"Circuit breaker for model {self.model_name} is {self.circuit_breaker.state.value}. "
                f"Request fast-rejected without network latency."
            )

        with self._lock:
            self.total_requests += 1

        # Check for simulated outage in mock/test mode
        if self._mock_failure:
            err_msg = f"Simulated 503/429 Outage on {self.model_name}"
            self._record_failure_metric(err_msg)
            self.circuit_breaker.record_failure(err_msg)
            raise RuntimeError(err_msg)

        headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        url = f"{HF_API_URL}/{self.model_name}"
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "return_full_text": False,
            },
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                latency = time.time() - t0

                # 503: Model loading
                if resp.status_code == 503:
                    wait_s = min(resp.json().get("estimated_time", 2.0), 3.0)
                    time.sleep(wait_s)
                    continue

                # 429: Rate limit
                if resp.status_code == 429:
                    if attempt < self.max_retries:
                        sleep_time = self.backoff_base_s * (2 ** attempt)
                        time.sleep(sleep_time)
                        continue
                    else:
                        raise RuntimeError(f"Rate limited (429) on {self.model_name}")

                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:150]}")

                result = resp.json()
                text = ""
                if isinstance(result, list) and result:
                    text = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    text = result.get("generated_text", "")

                # Record success
                self._record_success_metric(latency)
                self.circuit_breaker.record_success()
                return self._clean_response(text)

            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    sleep_time = self.backoff_base_s * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    break

        # If exhausted retries
        err_msg = str(last_exc) if last_exc else "Request failed after retries"
        self._record_failure_metric(err_msg)
        self.circuit_breaker.record_failure(err_msg)
        raise RuntimeError(f"Model {self.model_name} failed: {err_msg}")

    def _record_success_metric(self, latency_s: float):
        with self._lock:
            self.successful_requests += 1
            self.recent_latencies.append(latency_s)
            if len(self.recent_latencies) > 100:
                self.recent_latencies = self.recent_latencies[-100:]

    def _record_failure_metric(self, error_msg: str):
        with self._lock:
            self.failed_requests += 1
            self.last_error = error_msg
            self.last_error_time = time.time()

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            avg_lat = (sum(self.recent_latencies) / len(self.recent_latencies)) if self.recent_latencies else 0.0
            fail_rate = (self.failed_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0
            p95 = round(float(sorted(self.recent_latencies)[int(len(self.recent_latencies) * 0.95)]), 3) if len(self.recent_latencies) >= 5 else round(avg_lat, 3)
            return {
                "model_name": self.model_name,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "failure_rate_pct": round(fail_rate, 2),
                "avg_latency_s": round(avg_lat, 3),
                "p95_latency_s": p95,
                "circuit_breaker": self.circuit_breaker.get_status(),
                "last_error": self.last_error,
                "last_error_time": self.last_error_time,
            }

    @staticmethod
    def _clean_response(text: str) -> str:
        for token in ["<|end|>", "<|assistant|>", "<|user|>", "<|system|>", "<|eot_id|>"]:
            text = text.replace(token, "")
        return text.strip()


class LLMRouter:
    """
    Load-Balanced, Fault-Tolerant Router orchestrating multiple ModelEndpoints.
    Features:
    - Dynamic load balancing (Round-Robin, Least-Failures, Priority-Failover)
    - Zero-latency circuit-breaker failover
    - Aggregate and per-model telemetry
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        models: Optional[List[str]] = None,
        strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN,
        timeout_s: float = 6.0,
        max_retries: int = 2,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_s: float = 20.0,
    ):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.model_names = models or DEFAULT_MODELS
        self.strategy = strategy
        self.endpoints: Dict[str, ModelEndpoint] = {}

        for m in self.model_names:
            self.endpoints[m] = ModelEndpoint(
                model_name=m,
                hf_token=self.hf_token,
                timeout_s=timeout_s,
                max_retries=max_retries,
                circuit_failure_threshold=circuit_failure_threshold,
                circuit_cooldown_s=circuit_cooldown_s,
            )

        self._round_robin_idx: int = 0
        self._total_routed: int = 0
        self._total_failovers: int = 0
        self._lock = threading.Lock()

    def _select_candidate_endpoints(self) -> List[ModelEndpoint]:
        """
        Orders endpoints based on routing strategy and circuit breaker health.
        Healthy endpoints come first; open-circuit endpoints are placed at the end.
        """
        with self._lock:
            all_eps = [self.endpoints[m] for m in self.model_names]
            healthy = [ep for ep in all_eps if ep.circuit_breaker.can_attempt()]
            unhealthy = [ep for ep in all_eps if not ep.circuit_breaker.can_attempt()]

            if self.strategy == RoutingStrategy.ROUND_ROBIN:
                if healthy:
                    start = self._round_robin_idx % len(healthy)
                    self._round_robin_idx += 1
                    ordered_healthy = healthy[start:] + healthy[:start]
                else:
                    ordered_healthy = []
            elif self.strategy == RoutingStrategy.LEAST_FAILURES:
                ordered_healthy = sorted(healthy, key=lambda ep: ep.failed_requests)
            elif self.strategy == RoutingStrategy.LEAST_LATENCY:
                ordered_healthy = sorted(
                    healthy,
                    key=lambda ep: (sum(ep.recent_latencies) / len(ep.recent_latencies)) if ep.recent_latencies else 0.0
                )
            else:  # PRIORITY_FAILOVER (Qwen -> Mistral -> Zephyr)
                ordered_healthy = healthy

            return ordered_healthy + unhealthy

    def route_and_execute(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Tuple[str, str]:
        """
        Routes the prompt to the optimal healthy endpoint with automatic failover.
        Returns: Tuple[response_text, model_name_used]
        """
        with self._lock:
            self._total_routed += 1

        # Check for MOCK_LLM offline mode
        if os.environ.get("MOCK_LLM", "0") == "1":
            mock_resp, mock_model = self._execute_mock(system_prompt, user_prompt)
            return mock_resp, mock_model

        formatted_prompt = self._format_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        candidates = self._select_candidate_endpoints()
        last_error = None
        attempt_count = 0

        for ep in candidates:
            if not ep.circuit_breaker.can_attempt():
                continue

            try:
                attempt_count += 1
                response_text = ep.execute_request(
                    formatted_prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if attempt_count > 1:
                    with self._lock:
                        self._total_failovers += 1
                    logger.info(f"[Router] Failover successful! Handled by fallback: {ep.model_name}")

                return response_text, ep.model_name

            except Exception as exc:
                last_error = exc
                logger.warning(f"[Router] Model {ep.model_name} failed: {exc}. Shifting to next candidate.")
                continue

        # If all candidate endpoints failed or open
        logger.error(f"[Router] All endpoints exhausted or circuits OPEN. Last error: {last_error}")
        return f"Service currently undergoing high demand. Error: {last_error}", "none"

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Standard assistant string generation interface."""
        text, _ = self.route_and_execute(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return text

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> dict:
        """JSON assistant generation interface with parsing fallback."""
        if os.environ.get("MOCK_LLM", "0") == "1":
            return self._execute_mock_json(system_prompt, user_prompt)

        system_prompt = system_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no backticks, no extra text."
        raw, _ = self.route_and_execute(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return self._parse_json(raw)

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive router and endpoint telemetry."""
        with self._lock:
            ep_metrics = {name: ep.get_metrics() for name, ep in self.endpoints.items()}
            total_success = sum(m["successful_requests"] for m in ep_metrics.values())
            total_reqs = sum(m["total_requests"] for m in ep_metrics.values())
            success_rate = (total_success / total_reqs * 100) if total_reqs > 0 else 100.0

            return {
                "routing_strategy": self.strategy.value,
                "total_routed_requests": self._total_routed,
                "total_endpoint_invocations": total_reqs,
                "total_failovers": self._total_failovers,
                "overall_success_rate_pct": round(success_rate, 2),
                "endpoints": ep_metrics,
            }

    def simulate_endpoint_failure(self, model_name: str, enable_failure: bool = True):
        """Helper to simulate outages during resiliency & load testing."""
        if model_name in self.endpoints:
            self.endpoints[model_name]._mock_failure = enable_failure
            logger.info(f"[Router:Simulation] Mock failure set to {enable_failure} for {model_name}")

    def reset_all_circuits(self):
        """Reset all circuit breakers to healthy CLOSED state."""
        for ep in self.endpoints.values():
            ep.circuit_breaker.reset()
            ep._mock_failure = False

    @staticmethod
    def _format_messages(messages: list) -> str:
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

    @staticmethod
    def _parse_json(text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}

    def _execute_mock(self, system_prompt: str, user_prompt: str) -> Tuple[str, str]:
        candidates = self._select_candidate_endpoints()
        used_model = self.model_names[0]
        for ep in candidates:
            if not ep._mock_failure and ep.circuit_breaker.can_attempt():
                used_model = ep.model_name
                ep._record_success_metric(0.01)
                ep.circuit_breaker.record_success()
                break
            else:
                ep._record_failure_metric("Simulated outage")
                ep.circuit_breaker.record_failure("Simulated outage")

        if "evaluate" in system_prompt.lower() or "correctness" in system_prompt.lower():
            return "CORRECTNESS: Correct\n\nANALYSIS: The student answer is fully consistent with the syllabus.\n\nCORRECT ANSWER: Round Robin scheduling.\n\nIMPROVEMENT TIPS: Excellent understanding.", used_model
        if "is_grounded" in system_prompt.lower() or "fact-checking" in system_prompt.lower():
            return '{"is_grounded": true, "confidence_score": 0.95, "supported_claims": ["Verified against context"], "unsupported_claims": [], "contradictions": [], "citations": [{"claim": "Paging avoids external fragmentation", "context_snippet": "Paging eliminates external fragmentation"}], "reasoning": "High factual consistency"}', used_model
        if "mcq" in system_prompt.lower():
            return '{"question": "Which CPU scheduling algorithm allocates a fixed time slice per process?", "options": {"A": "Round Robin", "B": "First-Come First-Served", "C": "Shortest Job First", "D": "Priority Scheduling"}, "correct_option": "A", "explanation": "Round Robin assigns a time quantum to each process."}', used_model
        if "topic" in system_prompt.lower() or "syllabus" in system_prompt.lower():
            return '["Process Scheduling", "Memory Management", "Deadlocks", "Synchronization"]', used_model
        return "What is the primary role of Round Robin CPU scheduling in operating systems?", used_model

    def _execute_mock_json(self, system_prompt: str, user_prompt: str) -> dict:
        candidates = self._select_candidate_endpoints()
        for ep in candidates:
            if not ep._mock_failure and ep.circuit_breaker.can_attempt():
                ep._record_success_metric(0.01)
                ep.circuit_breaker.record_success()
                break
            else:
                ep._record_failure_metric("Simulated outage")
                ep.circuit_breaker.record_failure("Simulated outage")

        if "is_grounded" in system_prompt.lower() or "fact-checking" in system_prompt.lower():
            return {
                "is_grounded": True,
                "confidence_score": 0.95,
                "supported_claims": ["Verified against syllabus context."],
                "unsupported_claims": [],
                "contradictions": [],
                "citations": [{"claim": "Paging eliminates external fragmentation", "context_snippet": "Paging eliminates external fragmentation"}],
                "reasoning": "Full factual consistency verified.",
            }
        if "extract" in system_prompt.lower() or "extract" in user_prompt.lower() or "parsing" in system_prompt.lower():
            return ["Process Scheduling", "Memory Management", "Deadlocks", "Synchronization"]
        return {
            "question": "Which CPU scheduling algorithm allocates a fixed time slice per process in cyclic order?",
            "options": {
                "A": "Round Robin",
                "B": "First-Come First-Served",
                "C": "Shortest Job First",
                "D": "Priority Scheduling",
            },
            "correct_option": "A",
            "explanation": "Round Robin assigns a fixed time quantum to each ready process.",
        }
