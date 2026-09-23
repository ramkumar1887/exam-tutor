# 🎓 Exam Tutor AI — Grounded Agentic RAG with Automated Evaluation & Production Packaging

An adaptive, syllabus-grounded AI exam tutor built on **Grounded Agent Architecture**, **Automated Multi-Metric Evaluation Framework**, and **Production-Ready REST Service Packaging (FastAPI + Docker)**.

Focuses dynamically on what students **don't** know, eliminating hallucinations with syllabus entailment verification, and tracking topic mastery over time.

---

## 🌟 Key Architecture Pillars

### 1. 🛡️ Grounded Agent Architecture & Hallucination Elimination
- **FactVerificationAgent (`core/grounding.py`)**: Dedicated verification agent executing automated claim extraction, syllabus context entailment checks, and citation linking.
- **Refinement & Self-Correction Loops**: Automatic regenerating / grounding adjustment if claim confidence falls below $0.75$ or detects ungrounded contradictions.
- **Citation Attribution**: Every question premise and grading explanation returns grounding confidence and context citation quotes.

### 2. 📊 Automated Evaluation & Drift Framework (`eval/`)
- **Retrieval Faithfulness**: Stopword-filtered content token precision and morphological stem overlap against retrieved context chunks.
- **Structured JSON Schema Validity**: Strict Pydantic model validation (`MCQQuestionSchema`, `DescriptiveEvalSchema`) measuring structured generation reliability.
- **Semantic Uncertainty & Predictive Entropy**: Shannon entropy $H(X) = -\sum p(x) \log_2 p(x)$ assessing LLM decision ambiguity.
- **Confidence Calibration**: Brier Score ($BS = \frac{1}{N}\sum(f_t - o_t)^2$) and Expected Calibration Error (ECE) across multi-bin reliability diagrams.
- **Concept & Mastery Drift Detection**:
  - **Population Stability Index (PSI)**: Detects distribution shifts across historical and current mastery states.
  - **2-Sample Kolmogorov-Smirnov (KS) Test**: Evaluates whether score distributions have statistically drifted ($p < 0.05$).
- **Benchmark Suite**: Run automated evaluations via CLI (`python -m eval.benchmark`) or REST API (`/v1/eval/benchmark`).

### 3. 🚀 Modular Production Service Packaging (`api/` & `Docker`)
- **FastAPI REST Service (`api/main.py`)**:
  - `/v1/syllabus/upload-text`, `/v1/syllabus/upload-file`: Multi-format syllabus ingestion & TF-IDF indexing.
  - `/v1/sessions/*`: Stateful session management, topic mastery progress, and adaptive difficulty.
  - `/v1/sessions/{id}/next-question`: Fact-verified question generation.
  - `/v1/sessions/{id}/submit-answer`: Fact-checked grading and knowledge state updating.
  - `/v1/eval/benchmark`: On-demand automated evaluation harness runs.
  - `/v1/health`, `/v1/metrics`, `/v1/router/status`: Observability, router telemetry, and circuit breaker metrics.
- **Containerization**: Multi-stage `Dockerfile` and `docker-compose.yml` orchestrating both the FastAPI REST backend and Streamlit Web UI.

### 4. ⚡ Load-Balanced, Fault-Tolerant LLM Router (`core/router.py`)
- **Per-Model Circuit Breaker**: 3-state state machine (`CLOSED` -> `OPEN` -> `HALF_OPEN`) with configurable failure thresholds (e.g. 3 consecutive failures) and cooldown windows (e.g. 20s).
- **Dynamic Load Balancing**: Round-Robin, Least-Failures, and Priority-Failover strategies across `Qwen2.5-72B-Instruct`, `Mistral-7B-Instruct-v0.3`, and `zephyr-7b-beta`.
- **Zero-Latency Fast Rejection & Failover**: Immediately bypasses failing or open-circuit models without incurring network timeout stalls.
- **In-Memory Telemetry**: Real-time tracking of request counts, failure rates, EMA/P95 latencies, and total failovers.

---

## 📁 Project Structure

```
exam_tutor/
├── api/
│   ├── main.py                  # FastAPI RESTful application with versioned /v1 endpoints
│   └── schemas.py               # Pydantic request/response data contracts
├── core/
│   ├── router.py                # LLMRouter, ModelEndpoint, CircuitBreaker, and telemetry
│   ├── llm.py                   # LLMClient interface backed by LLMRouter
│   ├── grounding.py             # FactVerificationAgent, claim extraction, citation linking
│   ├── agents.py                # 4 LangGraph agents (assess, explain, quiz, progress)
│   ├── knowledge_tracker.py     # Per-topic mastery scores (SQLite + EMA tracking)
│   └── workflow.py              # LangGraph StateGraph wiring
├── eval/
│   ├── metrics.py               # Faithfulness, Schema Validity, Entropy, ECE/Brier, PSI & KS Drift
│   └── benchmark.py             # EvaluationHarness & CLI benchmark runner
├── rag/
│   └── pipeline.py              # Document chunking, TF-IDF vector index, topic extraction
├── ui/
│   └── app.py                   # Streamlit multi-page UI with radar charts, citations, & router telemetry
├── tests/
│   ├── run_tests.py             # End-to-end 10-part test suite runner
│   ├── test_router_concurrency.py # Router lifecycle & 20-worker concurrency failover tests
│   └── test_grounding_and_eval.py # Pytest automated test suite
├── Dockerfile                   # Multi-stage production container build
├── docker-compose.yml           # Multi-service orchestration (FastAPI + Streamlit)
├── requirements.txt             # Core dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Local Setup
```bash
# Clone and create virtualenv
git clone <repo-url>
cd exam_tutor
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Verification Tests & Benchmarks
```bash
# Run complete 9-part verification test suite (FactVerification, Metrics, Drift, API, Benchmarks)
python tests/run_tests.py

# Run pytest test suite
pytest tests/test_grounding_and_eval.py -v

# Run evaluation benchmark runner
python -m eval.benchmark --mock --rounds 4
```

### 3. Launch FastAPI REST Service
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation available at: `http://localhost:8000/docs`

### 4. Launch Streamlit Web UI
```bash
streamlit run ui/app.py
```
Access the application UI at: `http://localhost:8501`

### 5. Run with Docker Compose
```bash
docker compose up --build
```
- **REST API & Swagger Docs**: `http://localhost:8000/docs`
- **Streamlit Web UI**: `http://localhost:8501`

---

## 🧠 Adaptive Mastery & Study Loop

```
[Upload Syllabus (PDF/Text)]
            ↓
[RAG Ingestion: Chunking & Indexing]
            ↓
[Topic Extraction & Knowledge Tracker Initialization]
            ↓
┌───────────[ Adaptive Study Loop ]────────────────────────┐
│ 1. Prioritize weakest topics via Knowledge Tracker       │
│ 2. Retrieve syllabus context chunks (RAG)                │
│ 3. Generate adaptive MCQ / Descriptive question (LLM)    │
│ 4. FactVerificationAgent: entailment & citation check    │
│ 5. Evaluate user answer & provide syllabus-grounded tips │
│ 6. Update topic mastery score via Exponential Moving Avg │
│ 7. Adapt difficulty level (Easy → Medium → Hard)         │
└──────────────────────────────────────────────────────────┘
```

---

## 📈 Evaluation Metrics Summary

| Metric | Target | Formula / Method |
|---|---|---|
| **Retrieval Faithfulness** | $\ge 85\%$ | $\frac{\|T_{\text{gen}} \cap T_{\text{ctx}}\|}{\|T_{\text{gen}}\|}$ over content tokens & word stems |
| **JSON Schema Validity** | $100\%$ | Pydantic strict model validation (`MCQQuestionSchema`) |
| **Semantic Uncertainty** | Minimized | Shannon entropy $H(X) = -\sum p_i \log_2 p_i$ across answer samples |
| **Calibration (ECE & Brier)** | $\text{ECE} \le 0.10$ | Expected Calibration Error over binned confidence & Brier Score |
| **Topic Drift (PSI & KS)** | $\text{PSI} < 0.10$ | Population Stability Index & 2-sample Kolmogorov-Smirnov test |

---

## 🤖 LLM Strategy & Zero-Cost Architecture

- **Primary Model**: `Qwen/Qwen2.5-72B-Instruct` (Free tier on HuggingFace Inference API)
- **Fallback Models**: `mistralai/Mistral-7B-Instruct-v0.3`, `HuggingFaceH4/zephyr-7b-beta`
- **Offline / Mock Mode**: Set `MOCK_LLM=1` or pass `--mock` flag for zero-network testing and automated CI/CD pipelines.

---

## 📄 License
MIT License. Built with open-source tools for robust, grounded, and observable AI education systems.
