"""
CLI Test Runner — runs a mini tutor session in the terminal.
Useful for testing without Streamlit, or running in Google Colab.

Usage:
    python run_cli.py --topics "OS Basics,Process Scheduling,Memory Management" --rounds 5
    python run_cli.py --pdf path/to/syllabus.pdf --rounds 5
    HF_TOKEN=hf_xxx python run_cli.py --topics "Sorting Algorithms,Trees,Graphs"
"""

import os
import sys
import argparse
import uuid
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.llm import LLMClient
from core.knowledge_tracker import KnowledgeTracker
from core.agents import TutorAgents
from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING)


def run_cli(topics, session_name, n_rounds, pdf_path=None):
    print("\n" + "="*60)
    print("🎓 Exam Tutor AI — CLI Mode")
    print("="*60)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("⚠️  No HF_TOKEN set. LLM calls may fail. Set with: export HF_TOKEN=hf_xxx")

    llm = LLMClient(hf_token=token)
    rag = RAGPipeline(index_dir="data/uploads")
    tracker = KnowledgeTracker(db_path="data/sessions/knowledge.db")
    agents = TutorAgents(llm=llm, rag=rag, tracker=tracker)

    session_id = str(uuid.uuid4())

    # Ingest PDF if provided
    if pdf_path:
        print(f"📄 Ingesting PDF: {pdf_path}")
        chunks = rag.ingest_pdf(pdf_path)
        print(f"   → {len(chunks)} chunks extracted")
        if not topics:
            print("🧠 Extracting topics with AI...")
            raw = "\n".join(chunks)
            topics = rag.extract_topics_with_llm(llm, raw)
        rag.build_index(session_id)

    print(f"\n📋 Topics ({len(topics)}): {', '.join(topics[:5])}{'...' if len(topics) > 5 else ''}")
    print(f"🔁 Rounds: {n_rounds}")
    print("="*60)

    sess = tracker.create_session(session_id, session_name, topics)

    for rnd in range(1, n_rounds + 1):
        print(f"\n🧠 Round {rnd}/{n_rounds}")
        print(f"📌 Topic: {sess.current_topic or tracker.get_next_topic(sess)}")
        print("-"*50)

        # Generate question
        state = {"session": sess}
        print("Generating question...", end=" ", flush=True)
        qa = agents.assessment_agent(state)
        sess = qa["session"]
        print("done.")

        q_type = qa.get("question_type", "descriptive")
        print(f"\n❓ [{q_type.upper()}] {qa['question']}")

        if q_type == "mcq" and qa.get("mcq_options"):
            for k, v in qa["mcq_options"].items():
                print(f"   {k}: {v}")

        # Get user answer
        skip = input("\n⏭️  Skip (s) | Don't know (d) | Answer: ").strip()

        if skip.lower() == "s":
            sess = tracker.mark_topic_skipped(sess, qa["current_topic"])
            print("⏭️  Skipped.")
            continue

        if skip.lower() == "d":
            qa["user_answer"] = "I don't know"
        else:
            qa["user_answer"] = skip

        # Evaluate
        print("\n🤖 Evaluating...", end=" ", flush=True)
        feedback_result = agents.explainer_agent(qa)
        qa.update(feedback_result)
        print("done.")

        print(f"\n📝 Feedback ({qa.get('correctness_label', '')}):")
        print(qa.get("feedback", ""))

        # Progress
        progress_result = agents.progress_agent(qa)
        qa.update(progress_result)
        sess = progress_result["session"]

        print(f"\n📊 Score: {sess.total_correct}/{sess.total_questions} | Difficulty: {sess.difficulty_level}")

        # Practice question
        print("\nGenerating practice question...", end=" ", flush=True)
        pq = agents.quiz_agent(qa)
        print("done.")
        print(f"\n🧪 Practice: {pq.get('practice_question', '')}")
        if pq.get("practice_options"):
            for k, v in pq["practice_options"].items():
                mark = "✅" if k == pq.get("practice_correct") else "  "
                print(f"   {mark}{k}: {v}")

        input("\n▶️  Press Enter to continue...")

    print("\n" + "="*60)
    print("🏁 Session Complete!")
    summary = agents.get_session_summary(sess)
    print(summary)
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exam Tutor CLI")
    parser.add_argument("--topics", type=str, help="Comma-separated topics")
    parser.add_argument("--pdf", type=str, help="Path to syllabus PDF")
    parser.add_argument("--name", type=str, default="CLI Session", help="Session name")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    args = parser.parse_args()

    topics = [t.strip() for t in args.topics.split(",")] if args.topics else []

    if not topics and not args.pdf:
        # Interactive demo with sample topics
        topics = [
            "Process Scheduling",
            "Memory Management",
            "File Systems",
            "Deadlocks",
            "Virtual Memory",
            "CPU Architecture",
            "Semaphores and Mutex",
        ]
        print("No topics provided. Using OS sample topics.")

    os.makedirs("data/sessions", exist_ok=True)
    os.makedirs("data/uploads", exist_ok=True)

    run_cli(topics, args.name, args.rounds, pdf_path=args.pdf)
