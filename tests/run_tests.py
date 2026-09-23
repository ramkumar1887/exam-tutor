import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["MOCK_LLM"] = "1"
os.environ["TUTOR_DB_PATH"] = "data/sessions/test_knowledge.db"

from core.grounding import FactVerificationAgent
from core.knowledge_tracker import KnowledgeTracker
from eval.metrics import (
    compute_json_schema_validity,
    compute_retrieval_faithfulness,
    compute_semantic_uncertainty,
    compute_confidence_calibration,
    compute_population_stability_index,
    compute_ks_drift_test,
    MCQQuestionSchema,
)
from eval.benchmark import EvaluationHarness
from api.main import (
    health_check,
    get_service_metrics,
    ingest_text_syllabus,
    get_next_question,
    submit_answer,
    run_evaluation_benchmark,
)
from api.schemas import (
    IngestSyllabusRequest,
    SubmitAnswerRequest,
    BenchmarkRunRequest,
)

def run_all_tests():
    print("=" * 60)
    print(">> Running Exam Tutor AI Verification Test Suite...")
    print("=" * 60)

    # 1. Grounding tests
    print("1. Testing FactVerificationAgent...", end=" ", flush=True)
    verifier = FactVerificationAgent(llm_client=None)
    ctx = "Virtual memory uses paging to avoid external fragmentation. TLB speeds up page table lookups."
    cand = "Virtual memory implements paging to eliminate external fragmentation using a TLB."
    res = verifier.verify_and_refine(context=ctx, generated_text=cand, topic="Memory")
    assert res.is_grounded is True
    assert res.confidence_score > 0.60
    print("[PASSED]", flush=True)

    # 2. Schema validity
    print("2. Testing JSON Schema Validity metric...", end=" ", flush=True)
    valid_payload = {
        "question": "What is the primary function of a TLB?",
        "options": {"A": "Cache page table entries", "B": "Handle page replacement", "C": "Allocate stack memory", "D": "Schedule CPU threads"},
        "correct_option": "A",
        "explanation": "TLB caches virtual-to-physical address mappings.",
    }
    schema_res = compute_json_schema_validity([valid_payload], schema_cls=MCQQuestionSchema)
    assert schema_res["validity_rate"] == 1.0
    print("[PASSED]", flush=True)

    # 3. Retrieval Faithfulness
    print("3. Testing Retrieval Faithfulness metric...", end=" ", flush=True)
    ctx_f = ["A semaphore is a synchronization primitive with wait and signal operations."]
    gen_f = ["Semaphores are synchronization primitives that execute wait and signal."]
    faith_res = compute_retrieval_faithfulness(gen_f, ctx_f)
    assert faith_res["mean_faithfulness"] >= 0.80
    print(f"[PASSED] (score: {faith_res['mean_faithfulness']:.2%})", flush=True)

    # 4. Semantic Uncertainty & Calibration
    print("4. Testing Uncertainty and Calibration...", end=" ", flush=True)
    unc = compute_semantic_uncertainty(["A", "A", "A", "A"])
    assert unc["entropy"] == 0.0
    calib = compute_confidence_calibration([0.9, 0.85, 0.1], [True, True, False])
    assert calib["brier_score"] < 0.10
    print("[PASSED]", flush=True)

    # 5. Drift Detection (PSI & KS)
    print("5. Testing Drift Detectors (PSI & KS)...", end=" ", flush=True)
    baseline = [0.1, 0.2, 0.3, 0.4, 0.5]
    shifted = [0.85, 0.90, 0.92, 0.95, 0.99]
    psi = compute_population_stability_index(baseline, shifted)
    assert psi["is_drift_detected"] is True
    ks = compute_ks_drift_test(baseline, shifted)
    assert ks["drift_detected"] is True
    print("[PASSED]", flush=True)

    # 6. Knowledge Tracker & Adaptive Difficulty
    print("6. Testing Knowledge Tracker EMA & Adaptive Difficulty...", end=" ", flush=True)
    tracker = KnowledgeTracker(db_path="data/sessions/test_tracker.db")
    sess = tracker.create_session("sess_test", "OS", ["Processes"])
    sess = tracker.record_answer(sess, "Processes", True)
    assert sess.topics["Processes"].score == 0.3
    sess = tracker.record_answer(sess, "Processes", True)
    sess = tracker.record_answer(sess, "Processes", True)
    assert sess.difficulty_level == "hard"
    print("[PASSED]", flush=True)

    # 7. API Health & Metrics
    print("7. Testing API Health & Telemetry...", end=" ", flush=True)
    h = health_check()
    assert h["status"] == "healthy"
    m = get_service_metrics()
    assert "total_active_sessions" in m
    print("[PASSED]", flush=True)

    # 8. End-to-end Study Flow
    print("8. Testing End-to-End Syllabus & Study Loop Flow...", flush=True)
    upload_req = IngestSyllabusRequest(
        syllabus_name="Operating Systems",
        syllabus_text="Process Scheduling: Round Robin uses time slicing. Memory Management: Paging uses TLB.",
        topics=["Process Scheduling", "Memory Management"],
    )
    print("   -> Calling ingest_text_syllabus...", flush=True)
    up_resp = ingest_text_syllabus(upload_req)
    print(f"   -> Ingested session: {up_resp.session_id}", flush=True)

    print("   -> Calling get_next_question...", flush=True)
    q_resp = get_next_question(up_resp.session_id)
    print(f"   -> Question received: {q_resp.question[:30]}...", flush=True)
    assert q_resp.question != ""
    assert q_resp.is_grounded is True

    print("   -> Calling submit_answer...", flush=True)
    ans_req = SubmitAnswerRequest(
        user_answer="Round Robin",
        question=q_resp.question,
        question_type=q_resp.question_type,
    )
    ans_resp = submit_answer(up_resp.session_id, ans_req)
    print(f"   -> Answer submitted: is_correct={ans_resp.is_correct}", flush=True)
    assert ans_resp.is_correct is True
    assert ans_resp.session_state.total_questions == 1
    print("   -> Study Loop Flow: [PASSED]", flush=True)

    # 9. Automated Evaluation Benchmark Endpoint
    print("9. Testing Automated Evaluation Benchmark...", end=" ", flush=True)
    bench_req = BenchmarkRunRequest(num_rounds=2, topics=["Process Scheduling", "Memory Management"])
    b_report = run_evaluation_benchmark(bench_req)
    assert b_report["summary_metrics"]["task_completion_rate_pct"] == 100.0
    print("[PASSED]", flush=True)

    # 10. Load-Balanced LLM Router & Concurrency Failover
    print("10. Testing Load-Balanced LLM Router & Concurrency Failover...", end=" ", flush=True)
    from tests.test_router_concurrency import (
        test_circuit_breaker_lifecycle,
        test_round_robin_distribution,
        test_concurrent_stress_and_mid_run_outage,
    )
    test_circuit_breaker_lifecycle()
    test_round_robin_distribution()
    test_concurrent_stress_and_mid_run_outage()
    print("[PASSED]", flush=True)

    print("=" * 60)
    print("SUCCESS: ALL 10 VERIFICATION TEST SUITES PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
