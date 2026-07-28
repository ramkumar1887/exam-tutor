# 🎓 Exam Tutor AI

An adaptive study assistant that ingests your syllabus and quizzes you intelligently — focusing on what you **don't** know and skipping what you **do**.

Built on top of [NeoTutor](https://github.com/NeoTutor) (LangGraph agent loop) with:
- **RAG over your own syllabus PDF** (FAISS + sentence-transformers)
- **Per-topic knowledge state** (SQLite, persists across sessions)
- **Free LLMs via HuggingFace Inference API** (no local GPU needed)
- **Adaptive difficulty** (easy → medium → hard based on performance)
- **MCQ + Descriptive** question types
- **Streamlit UI** with topic progress tracking

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free HuggingFace token
Go to https://huggingface.co/settings/tokens → Create a **read** token (free).

### 3. Run the app
```bash
cd exam_tutor
streamlit run ui/app.py
```

### 4. Use the app
1. **Upload Syllabus** tab → upload your PDF or paste text
2. Enter your HuggingFace token in the sidebar
3. Switch to **Study Session** → start answering questions
4. Hit **⏭️ I know this, skip** on topics you're confident about
5. Check **Dashboard** for weak topics to revise

---

## 📁 Project Structure

```
exam_tutor/
├── core/
│   ├── llm.py              # HuggingFace Inference API wrapper (Qwen2.5, Mistral fallback)
│   ├── agents.py           # 4 LangGraph agents (assess, explain, quiz, progress)
│   ├── knowledge_tracker.py # Per-topic mastery scores (SQLite)
│   └── workflow.py         # LangGraph StateGraph wiring
├── rag/
│   └── pipeline.py         # PDF ingestion, chunking, FAISS index, topic extraction
├── ui/
│   └── app.py              # Streamlit multi-page UI
├── data/
│   ├── sessions/           # SQLite knowledge state DB
│   └── uploads/            # FAISS indexes per session
└── requirements.txt
```

---

## 🧠 How It Works

```
[Upload syllabus PDF]
       ↓
[RAG: chunk → embed → FAISS index]
       ↓
[LLM extracts topic list]
       ↓
[Knowledge Tracker initialises all topics at score=0]
       ↓
[Study Loop]
  ├── Pick weakest topic (lowest mastery score)
  ├── Retrieve relevant syllabus context (RAG)
  ├── Generate MCQ or descriptive question (LLM)
  ├── User answers (or skips "I know this")
  ├── Evaluate answer → give feedback (LLM)
  ├── Update topic score (exponential moving average)
  ├── Adapt difficulty (easy/medium/hard)
  └── Repeat
       ↓
[Session Summary: weak topics to revise]
```

---

## 🤖 LLMs Used (Free)

| Model | Used For |
|---|---|
| `Qwen/Qwen2.5-72B-Instruct` | Question generation, feedback (primary) |
| `mistralai/Mistral-7B-Instruct-v0.3` | Fallback if primary rate-limited |
| `HuggingFaceH4/zephyr-7b-beta` | Second fallback |
| `sentence-transformers/all-MiniLM-L6-v2` | Local embeddings (no API key) |

All LLM calls use HuggingFace Inference API free tier. If you hit rate limits, wait 60s.

---

## 🔧 Configuration

| Env Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | (set in UI) | HuggingFace read token |
| `TUTOR_DB_PATH` | `data/sessions/knowledge.db` | SQLite DB path |

---

## 📦 Credits

- **NeoTutor** — original LangGraph agent loop (`AIAgent_Tutor_System.ipynb`)
- **LangGraph** — agent state graph
- **sentence-transformers** — local embeddings
- **FAISS** — vector search
- **PyMuPDF** — PDF text extraction
- **Streamlit** — UI framework
