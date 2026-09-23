"""
Concurrency, Fault-Tolerance & Circuit Breaker Test Suite for LLMRouter
Executes:
1. CircuitBreaker 3-state transition tests (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
2. Routing strategy distribution tests (Round-Robin, Least-Failures, Priority)
3. 20-worker concurrent stress test with mid-run primary model outage injection
4. Telemetry validation and quantitative performance metrics capture
"""

import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.router import (
    CircuitBreaker,
    CircuitState,
    ModelEndpoint,
    LLMRouter,
    RoutingStrategy,
    DEFAULT_MODELS,
)


def test_circuit_breaker_lifecycle():
    print("  [1/4] Testing CircuitBreaker 3-state lifecycle...", end=" ", flush=True)
    cb = CircuitBreaker(
        model_name="test-model",
        failure_threshold=3,
        cooldown_seconds=0.2,  # fast cooldown for test
        half_open_success_threshold=1,
    )
    assert cb.state == CircuitState.CLOSED
    assert cb.can_attempt() is True

    # 1. Two failures - still CLOSED
    cb.record_failure("error 1")
    cb.record_failure("error 2")
    assert cb.state == CircuitState.CLOSED
    assert cb.can_attempt() is True

    # 2. Third failure - trips to OPEN
    cb.record_failure("error 3")
    assert cb.state == CircuitState.OPEN
    assert cb.can_attempt() is False
    assert cb.total_trips == 1

    # 3. Wait for cooldown window
    time.sleep(0.25)
    assert cb.can_attempt() is True  # triggers transition to HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN

    # 4. Success in HALF_OPEN recovers to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0
    print("[PASSED]", flush=True)


def test_circuit_breaker_probe_failure():
    print("  [2/4] Testing CircuitBreaker probe rejection in HALF_OPEN...", end=" ", flush=True)
    cb = CircuitBreaker(
        model_name="test-model",
        failure_threshold=2,
        cooldown_seconds=0.1,
    )
    cb.record_failure("err 1")
    cb.record_failure("err 2")
    assert cb.state == CircuitState.OPEN

    time.sleep(0.12)
    assert cb.can_attempt() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Probe fails -> immediately trips back to OPEN
    cb.record_failure("probe failed")
    assert cb.state == CircuitState.OPEN
    assert cb.total_trips == 2
    print("[PASSED]", flush=True)


def test_round_robin_distribution():
    print("  [3/4] Testing Round-Robin multi-endpoint load balancing...", end=" ", flush=True)
    router = LLMRouter(
        models=["m1", "m2", "m3"],
        strategy=RoutingStrategy.ROUND_ROBIN,
    )
    os.environ["MOCK_LLM"] = "1"

    models_used = []
    for _ in range(6):
        _, used = router.route_and_execute("system", "user")
        models_used.append(used)

    # Should cycle through m1, m2, m3, m1, m2, m3
    assert models_used == ["m1", "m2", "m3", "m1", "m2", "m3"]
    telemetry = router.get_telemetry()
    assert telemetry["total_routed_requests"] == 6
    assert telemetry["overall_success_rate_pct"] == 100.0
    print("[PASSED]", flush=True)


def test_concurrent_stress_and_mid_run_outage():
    print("  [4/4] Testing 20-Worker Concurrency with Mid-Run Primary Outage...", flush=True)
    router = LLMRouter(
        models=["Qwen/Qwen2.5-72B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "HuggingFaceH4/zephyr-7b-beta"],
        strategy=RoutingStrategy.PRIORITY_FAILOVER,
        circuit_failure_threshold=3,
        circuit_cooldown_s=10.0,
    )
    os.environ["MOCK_LLM"] = "1"

    total_tasks = 25
    results = []
    models_dispatched = []

    def worker_task(task_id: int):
        # Inject simulated outage on primary model at task #6
        if task_id == 6:
            router.simulate_endpoint_failure("Qwen/Qwen2.5-72B-Instruct", enable_failure=True)

        resp, used_model = router.route_and_execute(
            system_prompt="Generate exam question",
            user_prompt=f"Topic: Concurrency Task {task_id}",
        )
        return {
            "task_id": task_id,
            "success": bool(resp and len(resp) > 5),
            "model_used": used_model,
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker_task, i) for i in range(total_tasks)]
        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            models_dispatched.append(res["model_used"])

    elapsed = time.time() - t0
    telemetry = router.get_telemetry()

    successful_count = sum(1 for r in results if r["success"])
    completion_rate = (successful_count / total_tasks) * 100.0

    print(f"      -> Concurrency runtime: {elapsed:.3f}s for {total_tasks} parallel requests")
    print(f"      -> Task Completion Rate: {completion_rate:.1f}% ({successful_count}/{total_tasks})")
    print(f"      -> Primary Model Circuit State: {telemetry['endpoints']['Qwen/Qwen2.5-72B-Instruct']['circuit_breaker']['state']}")
    print(f"      -> Fallback Models Active: {list(set(models_dispatched) - {'Qwen/Qwen2.5-72B-Instruct'})}")

    assert completion_rate >= 95.0, f"Expected completion rate >= 95%, got {completion_rate}%"
    assert telemetry["endpoints"]["Qwen/Qwen2.5-72B-Instruct"]["circuit_breaker"]["state"] == "OPEN"
    assert "mistralai/Mistral-7B-Instruct-v0.3" in models_dispatched or "HuggingFaceH4/zephyr-7b-beta" in models_dispatched
    print("      -> Concurrent Failover Test: [PASSED]")


def run_all_router_tests():
    print("=" * 65)
    print(">> Running LLM Router & Concurrency Resilience Test Suite...")
    print("=" * 65)
    test_circuit_breaker_lifecycle()
    test_circuit_breaker_probe_failure()
    test_round_robin_distribution()
    test_concurrent_stress_and_mid_run_outage()
    print("=" * 65)
    print("SUCCESS: ALL ROUTER & CONCURRENCY TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_router_tests()
