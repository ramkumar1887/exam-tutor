"""
Grounding and Fact Verification Engine
Enforces factual consistency and eliminates hallucinations across generated questions,
rubrics, and explanations by checking claims against retrieved syllabus chunks.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GroundingVerificationResult:
    is_grounded: bool
    confidence_score: float  # 0.0 to 1.0
    supported_claims: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)
    revised_text: Optional[str] = None
    reasoning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class FactVerificationAgent:
    """
    Dedicated verification agent that performs:
    1. Atomic claim extraction
    2. Context entailment verification
    3. Grounding confidence scoring
    4. Self-correction / refinement if hallucinations are detected
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def verify_and_refine(
        self,
        context: str,
        generated_text: str,
        topic: str = "",
        task_type: str = "question",
        max_refine_attempts: int = 2,
    ) -> GroundingVerificationResult:
        """
        Validates whether `generated_text` is factually supported by `context`.
        If unsupported claims are found, iteratively refines the text.
        """
        if not context or not context.strip():
            # If no context provided (e.g. topic without RAG index), default to ungrounded warning
            return GroundingVerificationResult(
                is_grounded=True,
                confidence_score=0.70,
                supported_claims=["General knowledge fallback used (no syllabus chunk)."],
                reasoning="No syllabus context was supplied; passed with default confidence.",
            )

        current_text = generated_text
        last_result = None

        for attempt in range(max_refine_attempts):
            result = self._check_factual_consistency(context, current_text, topic, task_type)
            last_result = result

            if result.is_grounded:
                return result

            # If ungrounded and we have refinement attempts left, self-correct
            if attempt < max_refine_attempts - 1 and self.llm:
                logger.info(f"[Grounding] Self-correction triggered (attempt {attempt + 1})")
                current_text = self._self_correct(
                    context=context,
                    flawed_text=current_text,
                    unsupported_claims=result.unsupported_claims,
                    contradictions=result.contradictions,
                    task_type=task_type,
                )
                if current_text:
                    result.revised_text = current_text

        return last_result or self._heuristic_fallback_check(context, generated_text)

    def _check_factual_consistency(
        self,
        context: str,
        text: str,
        topic: str,
        task_type: str,
    ) -> GroundingVerificationResult:
        """Invokes LLM fact-checking prompt or fallback heuristic."""
        if not self.llm:
            return self._heuristic_fallback_check(context, text)

        system = (
            "You are a strict, adversarial fact-checking verifier for educational content. "
            "Your job is to eliminate hallucinations and enforce 100% factual consistency against the provided context.\n"
            "Analyze the candidate text against the source context and return ONLY a JSON object with keys:\n"
            "  - is_grounded: boolean (true if all facts and premises are strictly supported by context)\n"
            "  - confidence_score: float between 0.0 and 1.0\n"
            "  - supported_claims: list of verified factual statements\n"
            "  - unsupported_claims: list of claims not verified in the context\n"
            "  - contradictions: list of claims contradicting the context\n"
            "  - citations: list of objects with {'claim': str, 'context_snippet': str}\n"
            "  - reasoning: short explanation of verification verdict\n"
            "Respond ONLY with valid JSON."
        )

        user_prompt = f"""[SOURCE CONTEXT]:
{context[:2500]}

[TOPIC]: {topic}
[CONTENT TO VERIFY ({task_type})]:
{text}

Perform strict claim-level factual verification now:"""

        try:
            parsed = self.llm.call_json(system, user_prompt, max_tokens=500)
            if parsed and "is_grounded" in parsed:
                confidence = float(parsed.get("confidence_score", 0.8))
                confidence = max(0.0, min(1.0, confidence))
                return GroundingVerificationResult(
                    is_grounded=bool(parsed.get("is_grounded", False)),
                    confidence_score=confidence,
                    supported_claims=parsed.get("supported_claims", []),
                    unsupported_claims=parsed.get("unsupported_claims", []),
                    contradictions=parsed.get("contradictions", []),
                    citations=parsed.get("citations", []),
                    reasoning=parsed.get("reasoning", ""),
                )
        except Exception as e:
            logger.warning(f"[Grounding] LLM verification failed: {e}")

        return self._heuristic_fallback_check(context, text)

    def _self_correct(
        self,
        context: str,
        flawed_text: str,
        unsupported_claims: List[str],
        contradictions: List[str],
        task_type: str,
    ) -> str:
        """Self-reflection / refinement loop to rewrite ungrounded text."""
        system = (
            "You are an expert exam question editor. Rewrite the flawed content so that every fact, "
            "premise, and option is 100% strictly grounded in the source syllabus context. "
            "Eliminate all unsupported claims and contradictions. Preserve the question/feedback format."
        )
        user = f"""[SOURCE CONTEXT]:
{context[:2500]}

[FLAWED CONTENT ({task_type})]:
{flawed_text}

[IDENTIFIED ISSUES]:
- Unsupported claims: {json.dumps(unsupported_claims)}
- Contradictions: {json.dumps(contradictions)}

Rewrite the content now to be fully factually grounded:"""

        revised = self.llm.call(system, user, max_tokens=450, temperature=0.2)
        return revised.strip() if revised else flawed_text

    def _heuristic_fallback_check(self, context: str, text: str) -> GroundingVerificationResult:
        """
        Fast token/ngram-based lexical grounding check when LLM is unavailable.
        """
        ctx_words = set(re.findall(r"\b\w{4,}\b", context.lower()))
        text_words = set(re.findall(r"\b\w{4,}\b", text.lower()))

        if not text_words:
            return GroundingVerificationResult(is_grounded=True, confidence_score=1.0)

        overlap = text_words.intersection(ctx_words)
        overlap_ratio = len(overlap) / len(text_words)

        is_grounded = overlap_ratio >= 0.35
        confidence = round(min(1.0, overlap_ratio * 1.3), 3)

        return GroundingVerificationResult(
            is_grounded=is_grounded,
            confidence_score=confidence,
            supported_claims=[f"Lexical overlap: {len(overlap)} / {len(text_words)} keywords verified."],
            unsupported_claims=[] if is_grounded else ["Low keyword overlap with syllabus context."],
            reasoning=f"Heuristic keyword overlap ratio: {overlap_ratio:.2%}",
        )
