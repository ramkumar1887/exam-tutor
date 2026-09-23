"""
Unit & Integration Test Suite
Tests:
- FactVerificationAgent (Grounded agent architecture & hallucination detection)
- Automated Evaluation Metrics (Schema validity, Faithfulness, Uncertainty, ECE, Drift)
- KnowledgeTracker adaptive updates
- FastAPI REST Endpoints via TestClient
"""

import os
os.environ["MOCK_LLM"] = "1"
os.environ["TUTOR_DB_PATH"] = "data/sessions/test_knowledge.db"

import pytest
import numpy as np
from fastapi.testclient import TestClient

from core.grounding import FactVerificationAgent, GroundingVerificationResult
from core.knowledge_tracker import KnowledgeTracker, SessionState, TopicState
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
from api.main import app


# ──────────────────────────────────────────────────────────────
# 1. Grounding & Fact Verification Tests
# ──────────────────────────────────────────────────────────────

def test_fact_verification_grounded_heuristic():
    verifier = FactVerificationAgent(llm_client=None)
    context = "Virtual memory uses paging to avoid external fragmentation. TLB speeds up page table lookups."
    candidate = "Virtual memory implements paging to eliminate external fragmentation using a TLB."

    result = verifier.verify_and_refine(context=context, generated_text=candidate, topic="Memory")
    assert result.is_grounded is True
    assert result.confidence_score > 0.60
    assert len(result.supported_claims) > 0


def test_fact_verification_ungrounded_heuristic():
    verifier = FactVerificationAgent(llm_client=None)
    context = "Process scheduling algorithms include FCFS and Round Robin."
    candidate = "Quantum entanglement governs superconducting qubit coherence times in cryostats."

    result = verifier.verify_and_refine(context=context, generated_text=candidate, topic="Physics")
    assert result.is_grounded is False
    assert result.confidence_score < 0.50
    assert len(result.unsupported_claims) > 0


# ──────────────────────────────────────────────────────────────
# 2. Evaluation Framework Metric Tests
# ──────────────────────────────────────────────────────────────

def test_json_schema_validity():
    valid_payload = {
        "question": "What is the primary function of a TLB?",
        "options": {
            "A": "Cache page table entries",
            "B": "Handle page replacement",
            "C": "Allocate stack memory",
            "D": "Schedule CPU threads",
        },
        "correct_option": "A",
        "explanation": "TLB caches virtual-to-physical address mappings.",
    }
    invalid_payload = {"question": "Incomplete payload"}

    res = compute_json_schema_validity([valid_payload, invalid_payload], schema_cls=MCQQuestionSchema)
    assert res["validity_rate"] == 0.5
    assert res["valid_count"] == 1
    assert res["total"] == 2


def test_retrieval_faithfulness():
    ctx = ["A semaphore is a synchronization primitive with wait and signal operations."]
    gen = ["Semaphores are synchronization primitives that execute wait and signal."]
    res = compute_retrieval_faithfulness(gen, ctx)

    assert res["mean_faithfulness"] >= 0.80
    assert len(res["scores"]) == 1


def test_semantic_uncertainty_entropy():
    # Complete agreement -> zero entropy
    unanimous = ["A", "A", "A", "A"]
    res_unanimous = compute_semantic_uncertainty(unanimous)
    assert res_unanimous["entropy"] == 0.0

    # Split distribution -> positive entropy
    split_samples = ["A", "B", "C", "D"]
    res_split = compute_semantic_uncertainty(split_samples)
    assert res_split["entropy"] == 2.0  # log2(4)


def test_confidence_calibration():
    confs = [0.9, 0.8, 0.85, 0.2, 0.1]
    correct = [True, True, True, False, False]
    calib = compute_confidence_calibration(confs, correct, n_bins=5)

    assert calib["brier_score"] < 0.10
    assert calib["ece"] < 0.25
    assert len(calib["calibration_bins"]) > 0


def test_population_stability_index_drift():
    baseline = [0.1, 0.2, 0.3, 0.4, 0.5]
    same_distribution = [0.12, 0.21, 0.29, 0.41, 0.52]
    psi_res = compute_population_stability_index(baseline, same_distribution)

    assert psi_res["drift_level"] == "negligible"
    assert psi_res["is_drift_detected"] is False

    shifted_distribution = [0.85, 0.90, 0.92, 0.95, 0.99]
    drift_psi = compute_population_stability_index(baseline, shifted_distribution)
    assert drift_psi["is_drift_detected"] is True


def test_ks_drift_test():
    sample_a = [0.1, 0.2, 0.15, 0.25, 0.18]
    sample_b = [0.85, 0.9, 0.95, 0.88, 0.92]
    ks_res = compute_ks_drift_test(sample_a, sample_b)
    assert ks_res["drift_detected"] is True


# ──────────────────────────────────────────────────────────────
# 3. Knowledge Tracker & Adaptive Difficulty Tests
# ──────────────────────────────────────────────────────────────

def test_knowledge_tracker_ema_and_difficulty(tmp_path):
    db_file = str(tmp_path / "test_knowledge.db")
    tracker = KnowledgeTracker(db_path=db_file)
    sess = tracker.create_session("sess_123", "OS", ["Processes", "Memory"])

    # Initial score
    assert sess.topics["Processes"].score == 0.0

    # Answer correctly
    sess = tracker.record_answer(sess, "Processes", True)
    assert sess.topics["Processes"].score == 0.3  # alpha = 0.3 -> 0.3
    assert sess.topics["Processes"].attempts == 1
    assert sess.topics["Processes"].correct == 1

    # Answer 3 consecutive correct -> upgrade difficulty
    sess = tracker.record_answer(sess, "Processes", True)
    sess = tracker.record_answer(sess, "Processes", True)
    assert sess.difficulty_level == "hard"


# ──────────────────────────────────────────────────────────────
# 4. FastAPI REST API Integration Tests
# ──────────────────────────────────────────────────────────────

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


def test_api_health_and_metrics():
    res = health_check()
    assert res["status"] == "healthy"

    res_metrics = get_service_metrics()
    assert "total_active_sessions" in res_metrics


def test_api_syllabus_and_study_flow():
    # 1. Ingest text syllabus
    upload_req = IngestSyllabusRequest(
        syllabus_name="Operating Systems",
        syllabus_text="Process Scheduling: Round Robin uses time slicing. Memory Management: Paging uses TLB.",
        topics=["Process Scheduling", "Memory Management"],
    )
    resp = ingest_text_syllabus(upload_req)
    assert resp.session_id is not None
    assert len(resp.topics) == 2

    # 2. Get next question
    q_resp = get_next_question(resp.session_id)
    assert q_resp.question != ""
    assert q_resp.is_grounded is True
    assert q_resp.grounding_confidence > 0.0

    # 3. Submit answer
    ans_req = SubmitAnswerRequest(
        user_answer="Round Robin",
        question=q_resp.question,
        question_type=q_resp.question_type,
    )
    ans_resp = submit_answer(resp.session_id, ans_req)
    assert ans_resp.is_correct is True
    assert ans_resp.feedback != ""
    assert ans_resp.session_state.total_questions == 1


def test_api_evaluation_benchmark_endpoint():
    req = BenchmarkRunRequest(
        num_rounds=2,
        topics=["Process Scheduling", "Memory Management"],
    )
    res = run_evaluation_benchmark(req)
    assert res["summary_metrics"] is not None
    assert res["summary_metrics"]["retrieval_faithfulness_pct"] >= 0.0
    assert res["summary_metrics"]["json_schema_validity_pct"] >= 0.0
    assert res["summary_metrics"]["task_completion_rate_pct"] == 100.0
