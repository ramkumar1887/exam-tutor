"""
Automated Evaluation Framework
Metrics for:
1. Retrieval Faithfulness & Grounding
2. Structured JSON Schema Validity
3. Semantic Uncertainty & Predictive Entropy
4. Confidence Calibration (ECE & Brier Score)
5. Concept & Topic Mastery Drift Detection (PSI & KS Tests)
"""

import math
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, ValidationError
from scipy import stats


# ──────────────────────────────────────────────────────────────
# 1. Pydantic Schemas for Schema Validity Evaluation
# ──────────────────────────────────────────────────────────────

class MCQOptionSchema(BaseModel):
    A: str
    B: str
    C: str
    D: str


class MCQQuestionSchema(BaseModel):
    question: str
    options: MCQOptionSchema
    correct_option: str
    explanation: str


class DescriptiveEvalSchema(BaseModel):
    correctness: str
    analysis: str
    correct_answer: str
    improvement_tips: str


# ──────────────────────────────────────────────────────────────
# 2. Schema Validity Metric
# ──────────────────────────────────────────────────────────────

def compute_json_schema_validity(raw_outputs: List[Any], schema_cls=MCQQuestionSchema) -> Dict[str, Any]:
    """
    Evaluates what percentage of LLM responses strictly conform to target Pydantic schema.
    """
    total = len(raw_outputs)
    if total == 0:
        return {"validity_rate": 0.0, "valid_count": 0, "total": 0, "errors": []}

    valid_count = 0
    errors = []

    for idx, item in enumerate(raw_outputs):
        try:
            if isinstance(item, str):
                item = json.loads(item)
            schema_cls.model_validate(item)
            valid_count += 1
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as e:
            errors.append({"index": idx, "error": str(e), "raw": str(item)[:150]})

    return {
        "validity_rate": round(valid_count / total, 4),
        "valid_count": valid_count,
        "total": total,
        "errors": errors[:5],
    }


# ──────────────────────────────────────────────────────────────
# 3. Retrieval Faithfulness & Grounding Metric
# ──────────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "don", "should", "now", "are", "was", "were", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
}


def compute_retrieval_faithfulness(
    generated_texts: List[str],
    contexts: List[str],
) -> Dict[str, Any]:
    """
    Calculates faithfulness score of generated texts against source context chunks
    using token-level precision over informative content words and stem overlap.
    """
    if not generated_texts or not contexts or len(generated_texts) != len(contexts):
        return {"mean_faithfulness": 0.0, "scores": []}

    scores = []
    for gen, ctx in zip(generated_texts, contexts):
        if not ctx.strip():
            scores.append(0.0)
            continue

        raw_gen_tokens = re.findall(r"\b[a-zA-Z]{3,}\b", gen.lower())
        raw_ctx_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", ctx.lower()))

        # Filter stopwords
        gen_tokens = [t for t in raw_gen_tokens if t not in STOPWORDS]
        ctx_tokens = {t for t in raw_ctx_tokens if t not in STOPWORDS}

        if not gen_tokens:
            scores.append(1.0)
            continue

        def is_grounded_token(token: str, ref_set: set) -> bool:
            if token in ref_set:
                return True
            stem = token[:-1] if token.endswith('s') else token
            if stem in ref_set or any(r.startswith(stem) or stem.startswith(r) for r in ref_set if len(r) >= 4):
                return True
            return False

        grounded_tokens = [t for t in gen_tokens if is_grounded_token(t, ctx_tokens)]
        precision = len(grounded_tokens) / len(gen_tokens)

        faithfulness = min(1.0, precision * 1.1)
        scores.append(round(faithfulness, 4))

    return {
        "mean_faithfulness": round(float(np.mean(scores)), 4) if scores else 0.0,
        "median_faithfulness": round(float(np.median(scores)), 4) if scores else 0.0,
        "min_faithfulness": round(float(np.min(scores)), 4) if scores else 0.0,
        "scores": scores,
    }


# ──────────────────────────────────────────────────────────────
# 4. Semantic Uncertainty & Predictive Entropy
# ──────────────────────────────────────────────────────────────

def compute_semantic_uncertainty(predictions: List[str]) -> Dict[str, Any]:
    """
    Given multiple sampled answers / categories for a question (e.g., at temperature > 0),
    estimates semantic uncertainty via Shannon entropy: H = - sum(p_i * log2(p_i)).
    """
    if not predictions:
        return {"entropy": 0.0, "normalized_uncertainty": 0.0, "cluster_distribution": {}}

    # Group predictions by normalized semantic string
    counts: Dict[str, int] = {}
    for p in predictions:
        norm = p.strip().lower()
        counts[norm] = counts.get(norm, 0) + 1

    total = len(predictions)
    probs = [cnt / total for cnt in counts.values()]

    # Shannon Entropy
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(total) if total > 1 else 1.0
    norm_uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "entropy": round(entropy, 4),
        "normalized_uncertainty": round(norm_uncertainty, 4),
        "cluster_distribution": {k: round(v / total, 3) for k, v in counts.items()},
    }


# ──────────────────────────────────────────────────────────────
# 5. Confidence Calibration (ECE & Brier Score)
# ──────────────────────────────────────────────────────────────

def compute_confidence_calibration(
    confidences: List[float],
    correctness: List[bool],
    n_bins: int = 5,
) -> Dict[str, Any]:
    """
    Computes:
    - Brier Score: Mean squared error of probability predictions
    - Expected Calibration Error (ECE): Weighted difference between confidence and empirical accuracy across bins
    """
    if not confidences or not correctness or len(confidences) != len(correctness):
        return {"brier_score": 0.0, "ece": 0.0, "calibration_bins": []}

    y_true = np.array([1.0 if c else 0.0 for c in correctness])
    y_prob = np.clip(np.array(confidences), 0.0, 1.0)

    # Brier score
    brier = float(np.mean((y_prob - y_true) ** 2))

    # ECE computation
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)

        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (bin_size / len(y_prob)) * abs(bin_acc - bin_conf)
            bin_details.append({
                "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "count": bin_size,
                "accuracy": round(bin_acc, 3),
                "confidence": round(bin_conf, 3),
                "gap": round(abs(bin_acc - bin_conf), 3),
            })

    return {
        "brier_score": round(brier, 4),
        "ece": round(float(ece), 4),
        "calibration_bins": bin_details,
    }


# ──────────────────────────────────────────────────────────────
# 6. Concept & Topic Mastery Drift Detector
# ──────────────────────────────────────────────────────────────

def compute_population_stability_index(
    baseline_scores: List[float],
    current_scores: List[float],
    num_bins: int = 5,
) -> Dict[str, Any]:
    """
    Computes Population Stability Index (PSI) to detect concept/difficulty drift:
    PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
    - PSI < 0.1: No significant drift
    - 0.1 <= PSI < 0.25: Moderate drift
    - PSI >= 0.25: Significant drift / concept shift detected
    """
    if len(baseline_scores) < 2 or len(current_scores) < 2:
        return {"psi": 0.0, "drift_level": "insufficient_data"}

    b_arr = np.array(baseline_scores)
    c_arr = np.array(current_scores)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    b_counts, _ = np.histogram(b_arr, bins=bins)
    c_counts, _ = np.histogram(c_arr, bins=bins)

    # Smooth zero bins
    b_pct = (b_counts + 1e-5) / (len(b_arr) + 1e-5 * num_bins)
    c_pct = (c_counts + 1e-5) / (len(c_arr) + 1e-5 * num_bins)

    psi_val = float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))

    if psi_val < 0.10:
        level = "negligible"
    elif psi_val < 0.25:
        level = "moderate"
    else:
        level = "significant"

    return {
        "psi": round(psi_val, 4),
        "drift_level": level,
        "is_drift_detected": psi_val >= 0.10,
    }


def compute_ks_drift_test(
    sample_a: List[float],
    sample_b: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Two-sample Kolmogorov-Smirnov test to detect whether mastery score or difficulty distributions have drifted.
    """
    if len(sample_a) < 3 or len(sample_b) < 3:
        return {"p_value": 1.0, "statistic": 0.0, "drift_detected": False}

    ks_stat, p_val = stats.ks_2samp(sample_a, sample_b)
    return {
        "statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_val), 4),
        "drift_detected": bool(p_val < alpha),
    }
