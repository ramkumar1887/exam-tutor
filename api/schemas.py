"""
API Schemas for Exam Tutor REST Service
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class IngestSyllabusRequest(BaseModel):
    syllabus_text: Optional[str] = Field(None, description="Raw syllabus text or markdown")
    syllabus_name: str = Field("Course Syllabus", description="Title or name of the syllabus")
    topics: Optional[List[str]] = Field(None, description="Optional explicit topic list")


class IngestSyllabusResponse(BaseModel):
    session_id: str
    syllabus_name: str
    num_chunks: int
    topics: List[str]


class CreateSessionRequest(BaseModel):
    syllabus_name: str
    topics: List[str]
    session_id: Optional[str] = None


class SessionStateResponse(BaseModel):
    session_id: str
    syllabus_name: str
    current_topic: Optional[str]
    total_questions: int
    total_correct: int
    accuracy_pct: float
    difficulty_level: str
    weak_topics: List[str]
    mastered_topics: List[str]


class NextQuestionResponse(BaseModel):
    session_id: str
    topic: str
    question: str
    question_type: str  # mcq or descriptive
    options: Optional[Dict[str, str]] = None
    difficulty_level: str
    is_grounded: bool
    grounding_confidence: float
    citations: List[Dict[str, str]] = []


class SubmitAnswerRequest(BaseModel):
    user_answer: str
    question: Optional[str] = None
    question_type: Optional[str] = "mcq"
    mcq_correct: Optional[str] = ""
    mcq_options: Optional[Dict[str, str]] = None
    mcq_explanation: Optional[str] = ""


class SubmitAnswerResponse(BaseModel):
    is_correct: bool
    correctness_label: str
    feedback: str
    grounding: Dict[str, Any]
    session_state: SessionStateResponse


class PracticeQuestionResponse(BaseModel):
    question: str
    options: Dict[str, str]
    correct_option: str
    explanation: str


class BenchmarkRunRequest(BaseModel):
    num_rounds: int = Field(5, ge=1, le=20)
    syllabus_text: Optional[str] = None
    topics: Optional[List[str]] = None


class BenchmarkSummaryMetrics(BaseModel):
    retrieval_faithfulness_pct: float
    json_schema_validity_pct: float
    task_completion_rate_pct: float
    median_latency_s: float
    ece_calibration_error: float
    brier_score: float
    semantic_uncertainty_entropy: float
    psi_drift_level: str
    ks_drift_p_value: float


class BenchmarkRunResponse(BaseModel):
    timestamp: str
    session_id: str
    num_rounds_evaluated: int
    summary_metrics: BenchmarkSummaryMetrics
    detailed_reports: Dict[str, Any]
