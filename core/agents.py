"""
Tutor Agents (adapted from NeoTutor / AIAgent_Tutor_System.ipynb)
Four agents wired into a LangGraph workflow:
  1. assessment_agent  – generates a question on the current topic
  2. explainer_agent   – evaluates answer, gives structured feedback
  3. quiz_agent        – generates a follow-up MCQ practice question
  4. progress_agent    – updates scores, picks next topic

Key additions over NeoTutor:
  - RAG context injected into every question-generation prompt
  - Topic-aware (not just generic topic string)
  - Structured JSON output for MCQ options
  - "I know this" skip signal handled upstream in UI
"""

import json
import logging
import re
from typing import Dict, List, Optional

from core.llm import LLMClient
from core.knowledge_tracker import SessionState, KnowledgeTracker
from core.grounding import FactVerificationAgent, GroundingVerificationResult
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────

DIFFICULTY_INSTRUCTIONS = {
    "easy":   "Ask a basic factual/definition question. One correct answer is obvious.",
    "medium": "Ask a question requiring understanding and application. Require brief explanation.",
    "hard":   "Ask an advanced question requiring synthesis, comparison, or analysis.",
}


def _recent_q_context(state: SessionState) -> str:
    recent = state.question_history[-4:]
    if not recent:
        return ""
    return "Avoid repeating these recent questions:\n" + "\n".join(f"- {q}" for q in recent)


# ──────────────────────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────────────────────

class TutorAgents:
    """
    All four agents in one class.
    Each agent takes a SessionState dict-like and returns updates.
    Compatible with LangGraph StateGraph nodes.
    """

    def __init__(
        self,
        llm: LLMClient,
        rag: RAGPipeline,
        tracker: KnowledgeTracker,
        verifier: Optional[FactVerificationAgent] = None,
    ):
        self.llm = llm
        self.rag = rag
        self.tracker = tracker
        self.verifier = verifier or FactVerificationAgent(llm_client=llm)

    # ──────────────────────────────────────────────
    # 1. Assessment agent – generates the question
    # ──────────────────────────────────────────────

    def assessment_agent(self, state: dict) -> dict:
        """
        Generates either a descriptive or MCQ question for the current topic.
        Injects RAG context and runs strict factual grounding verification to eliminate hallucinations.
        """
        sess: SessionState = state["session"]
        topic = sess.current_topic or self.tracker.get_next_topic(sess)
        sess.current_topic = topic

        context = self.rag.get_context_for_topic(topic)
        difficulty = sess.difficulty_level
        diff_instruction = DIFFICULTY_INSTRUCTIONS.get(difficulty, DIFFICULTY_INSTRUCTIONS["medium"])
        recent_ctx = _recent_q_context(sess)

        # Alternate between descriptive and MCQ
        q_count = sess.total_questions
        use_mcq = (q_count % 3 != 0)  # 2 out of 3 questions are MCQ

        question_text = ""
        question_type = "descriptive"
        mcq_options = {}
        mcq_correct = ""
        mcq_explanation = ""

        if use_mcq:
            system = (
                "You are an expert exam question writer. "
                "Generate a multiple-choice question (MCQ) with exactly 4 options (A, B, C, D). "
                "CRITICAL: All premises, questions, and options MUST be factually grounded in the syllabus context. "
                "Return ONLY valid JSON with keys: question, options (object with A/B/C/D), correct_option (A/B/C/D), explanation. "
                "No markdown, no extra text."
            )
            user = f"""Topic: {topic}
Difficulty: {difficulty} — {diff_instruction}

Syllabus context:
{context if context else '(No specific context — use your knowledge)'}

{recent_ctx}

Generate an MCQ JSON now."""
            raw = self.llm.call_json(system, user, max_tokens=400)
            if raw and "question" in raw:
                question_text = raw.get("question", "")
                question_type = "mcq"
                mcq_options = raw.get("options", {})
                mcq_correct = raw.get("correct_option", "")
                mcq_explanation = raw.get("explanation", "")

        if not question_text:
            # Descriptive question fallback or default
            system = (
                "You are an expert exam tutor. Generate one clear exam question strictly grounded in the syllabus context. "
                "Return ONLY the question text, nothing else."
            )
            user = f"""Topic: {topic}
Difficulty: {difficulty} — {diff_instruction}

Syllabus context:
{context if context else '(No specific context — use your knowledge)'}

{recent_ctx}

Write one exam question:"""

            question = self.llm.call(system, user, max_tokens=150)
            question_text = question.strip().lstrip("Q:").lstrip("Question:").strip()
            question_type = "descriptive"

        # Grounding & Fact Verification step to eliminate hallucinations
        verification = self.verifier.verify_and_refine(
            context=context,
            generated_text=f"{question_text}\nExplanation: {mcq_explanation}",
            topic=topic,
            task_type="question_generation",
        )

        return {
            "session": sess,
            "question": question_text,
            "question_type": question_type,
            "mcq_options": mcq_options,
            "mcq_correct": mcq_correct,
            "mcq_explanation": mcq_explanation,
            "current_topic": topic,
            "grounding": verification.to_dict(),
            "is_grounded": verification.is_grounded,
            "grounding_confidence": verification.confidence_score,
            "citations": verification.citations,
        }

    # ──────────────────────────────────────────────
    # 2. Explainer agent – evaluates and gives feedback
    # ──────────────────────────────────────────────

    def explainer_agent(self, state: dict) -> dict:
        """
        Evaluates the user's answer and returns structured feedback.
        Runs factual consistency check against syllabus context to ensure no hallucinated corrections.
        """
        sess: SessionState = state["session"]
        question = state.get("question", "")
        user_answer = state.get("user_answer", "")
        q_type = state.get("question_type", "descriptive")
        topic = state.get("current_topic", sess.current_topic or "")

        # Get syllabus context for grounding the feedback
        context = self.rag.get_context_for_topic(topic)

        if q_type == "mcq":
            correct = state.get("mcq_correct", "").upper().strip()
            chosen = user_answer.upper().strip()
            is_correct = chosen == correct
            options = state.get("mcq_options", {})
            explanation = state.get("mcq_explanation", "")

            options_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
            feedback = (
                f"{'✅ Correct!' if is_correct else f'❌ Incorrect. The correct answer is **{correct}**.'}\n\n"
                f"**Options were:**\n{options_str}\n\n"
                f"**Explanation:** {explanation}"
            )

            verification = self.verifier.verify_and_refine(
                context=context,
                generated_text=feedback,
                topic=topic,
                task_type="mcq_feedback",
            )

            return {
                "session": sess,
                "feedback": feedback,
                "is_correct": is_correct,
                "correctness_label": "Correct" if is_correct else "Incorrect",
                "grounding": verification.to_dict(),
                "is_grounded": verification.is_grounded,
                "grounding_confidence": verification.confidence_score,
            }

        # Descriptive evaluation
        system = (
            "You are a patient, detailed exam tutor. "
            "Evaluate the student's answer strictly based on the syllabus context. "
            "Respond in this EXACT format:\n\n"
            "CORRECTNESS: [Correct / Partially Correct / Incorrect]\n\n"
            "ANALYSIS: [2-3 sentences on what's right or missing]\n\n"
            "CORRECT ANSWER: [The complete correct answer grounded in context]\n\n"
            "IMPROVEMENT TIPS: [1-2 actionable tips]"
        )
        user = f"""Question: {question}

Student's Answer: {user_answer}

Relevant syllabus context:
{context if context else '(Use your knowledge)'}

Evaluate the answer now:"""

        feedback = self.llm.call(system, user, max_tokens=400)

        # Parse correctness from feedback
        is_correct = False
        correctness_label = "Incorrect"
        feedback_lower = feedback.lower()
        if re.search(r"correctness:\s*correct\b", feedback_lower):
            is_correct = True
            correctness_label = "Correct"
        elif re.search(r"correctness:\s*partially", feedback_lower):
            is_correct = True  # give partial credit
            correctness_label = "Partially Correct"

        verification = self.verifier.verify_and_refine(
            context=context,
            generated_text=feedback,
            topic=topic,
            task_type="descriptive_evaluation",
        )

        return {
            "session": sess,
            "feedback": feedback,
            "is_correct": is_correct,
            "correctness_label": correctness_label,
            "grounding": verification.to_dict(),
            "is_grounded": verification.is_grounded,
            "grounding_confidence": verification.confidence_score,
        }

    # ──────────────────────────────────────────────
    # 3. Quiz agent – generates a follow-up practice MCQ
    # ──────────────────────────────────────────────

    def quiz_agent(self, state: dict) -> dict:
        """
        After feedback, generate a quick follow-up practice question.
        Always MCQ for fast reinforcement.
        """
        sess: SessionState = state["session"]
        topic = state.get("current_topic", sess.current_topic or "")
        prev_q = state.get("question", "")
        context = self.rag.get_context_for_topic(topic)

        system = (
            "You are an exam quiz generator. Generate a short MCQ for quick practice grounded in the context. "
            "Return ONLY valid JSON with: question, options (A/B/C/D), correct_option, explanation."
        )
        user = f"""Topic: {topic}
Previous question (avoid repeating): {prev_q}

Context:
{context if context else '(Use your knowledge)'}

Generate a different, simpler reinforcement MCQ:"""

        raw = self.llm.call_json(system, user, max_tokens=350)

        if raw and "question" in raw:
            p_q = raw.get("question", "")
            p_opts = raw.get("options", {})
            p_corr = raw.get("correct_option", "")
            p_exp = raw.get("explanation", "")

            verification = self.verifier.verify_and_refine(
                context=context,
                generated_text=f"{p_q}\n{p_exp}",
                topic=topic,
                task_type="practice_mcq",
            )

            return {
                "session": sess,
                "practice_question": p_q,
                "practice_options": p_opts,
                "practice_correct": p_corr,
                "practice_explanation": p_exp,
                "grounding": verification.to_dict(),
            }

        # Fallback plain question
        fallback = self.llm.call(
            "Generate one quick practice question. Return only the question.",
            f"Topic: {topic}. Different from: {prev_q}",
            max_tokens=100,
        )
        return {
            "session": sess,
            "practice_question": fallback.strip(),
            "practice_options": {},
            "practice_correct": "",
            "practice_explanation": "",
        }

    # ──────────────────────────────────────────────
    # 4. Progress agent – updates tracker, picks next topic
    # ──────────────────────────────────────────────

    def progress_agent(self, state: dict) -> dict:
        """
        Records the answer, updates knowledge state, picks next topic.
        """
        sess: SessionState = state["session"]
        topic = state.get("current_topic", sess.current_topic or "")
        is_correct: bool = state.get("is_correct", False)
        question = state.get("question", "")

        # Record to knowledge tracker
        sess = self.tracker.record_answer(sess, topic, is_correct)

        # Add question to history to avoid repeats
        if question and question not in sess.question_history:
            sess.question_history.append(question)
            if len(sess.question_history) > 20:
                sess.question_history = sess.question_history[-20:]

        # Pick next topic
        next_topic = self.tracker.get_next_topic(sess)
        sess.current_topic = next_topic

        self.tracker.save_session(sess)

        return {
            "session": sess,
            "next_topic": next_topic,
            "score": sess.total_correct,
            "total": sess.total_questions,
            "difficulty_level": sess.difficulty_level,
        }

    # ──────────────────────────────────────────────
    # Summary helper
    # ──────────────────────────────────────────────

    def get_session_summary(self, sess: SessionState) -> str:
        if sess.total_questions == 0:
            return "No questions answered yet."

        acc = sess.total_correct / sess.total_questions * 100
        weak = self.tracker.get_weak_topics(sess, n=5)
        mastered = [
            ts.topic for ts in sess.topics.values()
            if ts.score >= 0.75 and ts.attempts >= 2
        ]

        lines = [
            f"**Questions answered:** {sess.total_questions}",
            f"**Correct:** {sess.total_correct} ({acc:.1f}%)",
            f"**Current difficulty:** {sess.difficulty_level.title()}",
            "",
            f"**Topics to revise:** {', '.join(weak) if weak else 'None — great job!'}",
            f"**Topics mastered:** {', '.join(mastered) if mastered else 'Keep going!'}",
        ]
        return "\n".join(lines)
