"""
FastAPI Production REST Service for Exam Tutor AI
Provides scalable APIs for:
- Syllabus ingestion & RAG indexing
- Stateful multi-agent study sessions with adaptive difficulty
- Grounded question generation & fact-verified answer grading
- Automated evaluation and benchmarking endpoints
"""

import os
import sys
import uuid
import time
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm import LLMClient
from core.knowledge_tracker import KnowledgeTracker, SessionState
from core.grounding import FactVerificationAgent
from core.agents import TutorAgents
from rag.pipeline import RAGPipeline
from eval.benchmark import EvaluationHarness
from api.schemas import (
    IngestSyllabusRequest,
    IngestSyllabusResponse,
    CreateSessionRequest,
    SessionStateResponse,
    NextQuestionResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    PracticeQuestionResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
)

# App Configuration
APP_TITLE = "Exam Tutor AI — Production API"
APP_VERSION = "2.0.0"

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Production REST API for Adaptive RAG-Powered AI Exam Tutor with Grounded Agent Architecture & Automated Evaluation Framework",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Service Singletons / Dependencies
_tracker = KnowledgeTracker(db_path=os.environ.get("TUTOR_DB_PATH", "data/sessions/knowledge.db"))
_rag = RAGPipeline(index_dir="data/uploads")
_llm = LLMClient(hf_token=os.environ.get("HF_TOKEN", ""))
_verifier = FactVerificationAgent(llm_client=_llm)
_agents = TutorAgents(llm=_llm, rag=_rag, tracker=_tracker, verifier=_verifier)
_harness = EvaluationHarness(rag=_rag, llm=_llm, verifier=_verifier)


def _build_session_response(sess: SessionState) -> SessionStateResponse:
    total = sess.total_questions
    corr = sess.total_correct
    acc = round((corr / total * 100), 2) if total > 0 else 0.0
    weak = _tracker.get_weak_topics(sess, n=5)
    mastered = [
        ts.topic for ts in sess.topics.values()
        if ts.score >= 0.75 and ts.attempts >= 2
    ]
    return SessionStateResponse(
        session_id=sess.session_id,
        syllabus_name=sess.syllabus_name,
        current_topic=sess.current_topic,
        total_questions=total,
        total_correct=corr,
        accuracy_pct=acc,
        difficulty_level=sess.difficulty_level,
        weak_topics=weak,
        mastered_topics=mastered,
    )


# ──────────────────────────────────────────────────────────────
# Health & Observability Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/v1/health", tags=["Observability"])
def health_check():
    """Service health and component status."""
    return {
        "status": "healthy",
        "service": APP_TITLE,
        "version": APP_VERSION,
        "llm_model": _llm._get_model(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.get("/v1/metrics", tags=["Observability"])
def get_service_metrics():
    """Aggregated service telemetry, knowledge session statistics, and LLM router metrics."""
    sessions = _tracker.list_sessions()
    return {
        "total_active_sessions": len(sessions),
        "llm_active_model": _llm._get_model(),
        "storage_db_path": _tracker.db_path,
        "rag_indices_available": len(list(Path("data/uploads").glob("*.tfidf.pkl"))),
        "router_telemetry": _llm.get_telemetry(),
    }


@app.get("/v1/router/status", tags=["Observability"])
def get_router_status():
    """Real-time load balancer status, circuit breaker states, and per-endpoint latency/failure stats."""
    return _llm.get_telemetry()


# ──────────────────────────────────────────────────────────────
# Syllabus & Ingestion Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/v1/syllabus/upload-text", response_model=IngestSyllabusResponse, tags=["Syllabus"])
def ingest_text_syllabus(payload: IngestSyllabusRequest):
    """Ingest raw syllabus text, extract testable topics, and build RAG index."""
    if not payload.syllabus_text:
        raise HTTPException(status_code=400, detail="syllabus_text is required.")

    session_id = str(uuid.uuid4())
    chunks = _rag.ingest_text(payload.syllabus_text, source=payload.syllabus_name)
    _rag.build_index(session_id)

    topics = payload.topics
    if not topics:
        topics = _rag.extract_topics_with_llm(_llm, payload.syllabus_text)
    if not topics:
        topics = ["General Topic 1", "General Topic 2"]

    _tracker.create_session(session_id, payload.syllabus_name, topics)

    return IngestSyllabusResponse(
        session_id=session_id,
        syllabus_name=payload.syllabus_name,
        num_chunks=len(chunks),
        topics=topics,
    )


@app.post("/v1/syllabus/upload-file", response_model=IngestSyllabusResponse, tags=["Syllabus"])
async def ingest_file_syllabus(
    file: UploadFile = File(...),
    syllabus_name: Optional[str] = Form("Uploaded Syllabus"),
):
    """Upload PDF syllabus document, parse with PyMuPDF, and index."""
    session_id = str(uuid.uuid4())
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"{session_id}_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    if file.filename.endswith(".pdf"):
        chunks = _rag.ingest_pdf(str(temp_path))
    else:
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = _rag.ingest_text(text, source=syllabus_name)

    _rag.build_index(session_id)

    raw_sample = "\n".join(chunks[:5])
    topics = _rag.extract_topics_with_llm(_llm, raw_sample)
    if not topics:
        topics = ["General Overview", "Core Concepts"]

    _tracker.create_session(session_id, syllabus_name, topics)

    return IngestSyllabusResponse(
        session_id=session_id,
        syllabus_name=syllabus_name,
        num_chunks=len(chunks),
        topics=topics,
    )


# ──────────────────────────────────────────────────────────────
# Session & Question Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/v1/sessions", response_model=SessionStateResponse, tags=["Sessions"])
def create_or_resume_session(req: CreateSessionRequest):
    """Create a new study session or initialize topic mastery states."""
    session_id = req.session_id or str(uuid.uuid4())
    sess = _tracker.load_session(session_id)
    if not sess:
        sess = _tracker.create_session(session_id, req.syllabus_name, req.topics)
    return _build_session_response(sess)


@app.get("/v1/sessions/{session_id}/state", response_model=SessionStateResponse, tags=["Sessions"])
def get_session_state(session_id: str):
    """Retrieve full knowledge tracker mastery state for a session."""
    sess = _tracker.load_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return _build_session_response(sess)


@app.get("/v1/sessions/{session_id}/next-question", response_model=NextQuestionResponse, tags=["Study Loop"])
def get_next_question(session_id: str):
    """
    Generate an adaptively selected, syllabus-grounded question with hallucination verification.
    """
    sess = _tracker.load_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    # Ensure index is loaded if exists
    _rag.load_index(session_id)

    state = {"session": sess}
    qa = _agents.assessment_agent(state)
    _tracker.save_session(qa["session"])

    return NextQuestionResponse(
        session_id=session_id,
        topic=qa["current_topic"],
        question=qa["question"],
        question_type=qa["question_type"],
        options=qa.get("mcq_options"),
        difficulty_level=qa["session"].difficulty_level,
        is_grounded=qa.get("is_grounded", True),
        grounding_confidence=qa.get("grounding_confidence", 1.0),
        citations=qa.get("citations", []),
    )


@app.post("/v1/sessions/{session_id}/submit-answer", response_model=SubmitAnswerResponse, tags=["Study Loop"])
def submit_answer(session_id: str, req: SubmitAnswerRequest):
    """
    Evaluate user answer, run fact-checking on grading explanation, and update knowledge tracker.
    """
    sess = _tracker.load_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    _rag.load_index(session_id)

    state = {
        "session": sess,
        "user_answer": req.user_answer,
        "question": req.question or "",
        "question_type": req.question_type or "mcq",
        "mcq_correct": req.mcq_correct or "",
        "mcq_options": req.mcq_options or {},
        "mcq_explanation": req.mcq_explanation or "",
    }

    # Evaluate
    exp_res = _agents.explainer_agent(state)
    state.update(exp_res)

    # Update progress & state
    prog_res = _agents.progress_agent(state)
    updated_sess = prog_res["session"]

    return SubmitAnswerResponse(
        is_correct=exp_res["is_correct"],
        correctness_label=exp_res["correctness_label"],
        feedback=exp_res["feedback"],
        grounding=exp_res.get("grounding", {}),
        session_state=_build_session_response(updated_sess),
    )


# ──────────────────────────────────────────────────────────────
# Automated Evaluation Framework Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/v1/eval/benchmark", response_model=BenchmarkRunResponse, tags=["Evaluation Framework"])
def run_evaluation_benchmark(req: BenchmarkRunRequest):
    """
    Run automated evaluation benchmark measuring faithfulness, schema validity,
    confidence calibration, predictive entropy, and drift detection.
    """
    syllabus_content = req.syllabus_text if req.syllabus_text else None
    report = _harness.run_benchmark(
        syllabus_text=syllabus_content or None,
        topics=req.topics,
        num_rounds=req.num_rounds,
    )
    return report
