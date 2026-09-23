"""
Exam Tutor – Streamlit UI
Run with: streamlit run ui/app.py
"""

import os
import sys
import uuid
import logging
import tempfile
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm import LLMClient
from core.knowledge_tracker import KnowledgeTracker, SessionState
from core.agents import TutorAgents
from rag.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Exam Tutor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
.topic-badge {
    display: inline-block;
    background: #1e3a5f;
    color: white;
    border-radius: 12px;
    padding: 3px 10px;
    margin: 3px;
    font-size: 0.8rem;
}
.weak-badge {
    background: #7f1d1d;
}
.strong-badge {
    background: #14532d;
}
.score-card {
    background: #0f172a;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid #334155;
}
.question-box {
    background: #1e293b;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1.2rem;
    margin: 1rem 0;
    font-size: 1.05rem;
}
.feedback-correct {
    background: #052e16;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 1rem;
}
.feedback-partial {
    background: #1c1917;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 1rem;
}
.feedback-wrong {
    background: #2d0a0a;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Initialise singletons in session state
# ──────────────────────────────────────────────

@st.cache_resource
def get_tracker():
    return KnowledgeTracker(db_path="data/sessions/knowledge.db")

@st.cache_resource
def get_llm():
    token = os.environ.get("HF_TOKEN", "")
    return LLMClient(hf_token=token)

def get_rag() -> RAGPipeline:
    if "rag" not in st.session_state:
        st.session_state.rag = RAGPipeline(index_dir="data/uploads")
    return st.session_state.rag

def get_agents() -> TutorAgents:
    if "agents" not in st.session_state:
        st.session_state.agents = TutorAgents(
            llm=get_llm(),
            rag=get_rag(),
            tracker=get_tracker(),
        )
    return st.session_state.agents

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    tracker = get_tracker()

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=60)
        st.title("Exam Tutor AI")
        st.caption("Adaptive study sessions powered by LLMs")

        st.divider()

        page = st.radio(
            "Navigate",
            ["📤 Upload Syllabus", "📚 Study Session", "📊 Dashboard"],
            key="page_nav",
        )

        st.divider()

        # HF Token input
        hf_token = st.text_input(
            "HuggingFace Token",
            value=os.environ.get("HF_TOKEN", ""),
            type="password",
            help="Get a free token at huggingface.co/settings/tokens",
        )
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            # Refresh LLM client token
            get_llm().token = hf_token
            get_llm().headers = {"Authorization": f"Bearer {hf_token}"}

        # Resume existing session
        st.divider()
        sessions = tracker.list_sessions()
        if sessions:
            st.subheader("Resume Session")
            session_labels = {f"{s[1]} ({s[0][:8]}...)": s[0] for s in sessions}
            chosen_label = st.selectbox("Pick session", ["— new —"] + list(session_labels.keys()))
            if chosen_label != "— new —" and st.button("Resume"):
                sid = session_labels[chosen_label]
                sess = tracker.load_session(sid)
                if sess:
                    st.session_state.current_session = sess
                    st.session_state.page_nav = "📚 Study Session"
                    st.session_state.qa_state = {}
                    st.rerun()

        # LLM Router & Circuit Breaker Telemetry Monitor
        st.divider()
        with st.expander("⚡ LLM Router & Circuit Breakers", expanded=False):
            telemetry = get_llm().get_telemetry()
            st.caption(f"Strategy: **{telemetry['routing_strategy']}** | Success: **{telemetry['overall_success_rate_pct']}%**")
            st.caption(f"Routed: **{telemetry['total_routed_requests']}** | Failovers: **{telemetry['total_failovers']}**")
            st.markdown("---")
            for mname, data in telemetry["endpoints"].items():
                short_name = mname.split("/")[-1]
                state = data["circuit_breaker"]["state"]
                state_color = "🟢" if state == "CLOSED" else ("🟡" if state == "HALF_OPEN" else "🔴")
                st.markdown(f"**{state_color} {short_name}**")
                st.caption(f"State: `{state}` | Latency: `{data['avg_latency_s']}s` | Req: `{data['successful_requests']}/{data['total_requests']}`")

    return page

# ──────────────────────────────────────────────
# Page: Upload Syllabus
# ──────────────────────────────────────────────

def page_upload():
    st.header("📤 Upload Syllabus")
    st.markdown("Upload your **syllabus PDF**, paste text, or enter topics manually. The AI will extract topics and build a study plan.")

    tab1, tab2, tab3 = st.tabs(["📄 Upload PDF", "📝 Paste Text", "✏️ Enter Topics"])

    raw_text = ""
    topics_override = []

    with tab1:
        uploaded = st.file_uploader("Upload syllabus PDF", type=["pdf"])
        if uploaded:
            # Only process if this is a NEW file (not a rerun)
            if st.session_state.get("last_uploaded") != uploaded.name:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                rag = get_rag()
                with st.spinner("Extracting text from PDF..."):
                    logger.info("[UI] Starting PDF ingest")
                    chunks = rag.ingest_pdf(tmp_path)
                    logger.info(f"[UI] Ingest done, {len(chunks)} chunks")
                    raw_text = "\n".join(chunks)
                    os.unlink(tmp_path)

                # Cache results so reruns don't reprocess
                st.session_state["last_uploaded"] = uploaded.name
                st.session_state["raw_text"] = raw_text
                st.session_state["chunks_count"] = len(chunks)

            raw_text = st.session_state.get("raw_text", "")
            st.success(f"✅ Extracted {st.session_state.get('chunks_count', 0)} text chunks from PDF")

    with tab2:
        pasted = st.text_area("Paste syllabus text here", height=200)
        if pasted.strip():
            raw_text = pasted

    with tab3:
        manual = st.text_area(
            "Enter topics one per line",
            placeholder="Operating System Basics\nProcess Scheduling\nMemory Management\n...",
            height=200,
        )
        if manual.strip():
            topics_override = [t.strip() for t in manual.strip().splitlines() if t.strip()]

    st.divider()

    syllabus_name = st.text_input("Session name", placeholder="e.g. OS Mid-sem 2025")
    custom_instructions = st.text_area(
        "Special instructions (optional)",
        placeholder="Focus more on Unit 3. Ask MCQs only. Ignore networking topics.",
        height=80,
    )

    if st.button("🚀 Create Study Session", type="primary"):
        if not syllabus_name:
            st.error("Please enter a session name.")
            return

        tracker = get_tracker()
        llm = get_llm()
        rag = get_rag()

        topics = topics_override

        if not topics and raw_text:
            with st.spinner("🧠 Extracting topics with AI..."):
                logger.info("[UI] Starting topic extraction")
                topics = rag.extract_topics_with_llm(llm, raw_text)
                logger.info(f"[UI] Topics extracted: {topics}")

        if not topics:
            st.error("Could not extract topics. Please use the 'Enter Topics' tab.")
            return

        session_id = str(uuid.uuid4())

        # Build RAG index if we have text
        if raw_text and not topics_override:
            with st.spinner("Building search index..."):
                logger.info("[UI] Building FAISS index")
                rag.build_index(session_id)
                logger.info("[UI] Index built")

        sess = tracker.create_session(session_id, syllabus_name, topics)
        if custom_instructions:
            sess.extra["instructions"] = custom_instructions
            tracker.save_session(sess)

        st.session_state.current_session = sess
        st.session_state.qa_state = {}
        st.session_state.page_nav = "📚 Study Session"

        st.success(f"✅ Session created with **{len(topics)} topics**!")
        st.markdown("**Topics extracted:**")
        cols = st.columns(3)
        for i, t in enumerate(topics):
            cols[i % 3].markdown(f'<span class="topic-badge">{t}</span>', unsafe_allow_html=True)

        st.info("👈 Switch to **Study Session** in the sidebar to begin!")

# ──────────────────────────────────────────────
# Page: Study Session
# ──────────────────────────────────────────────

def page_study():
    st.header("📚 Study Session")

    sess: SessionState = st.session_state.get("current_session")
    if sess is None:
        st.warning("No active session. Please upload a syllabus first.")
        return

    tracker = get_tracker()
    agents = get_agents()

    # Load RAG index if available
    rag = get_rag()
    if not rag._index and rag.has_index(sess.session_id):
        rag.load_index(sess.session_id)

    qa = st.session_state.get("qa_state", {})

    # ── Top metrics bar ──
    col1, col2, col3, col4 = st.columns(4)
    total = sess.total_questions
    correct = sess.total_correct
    col1.metric("Questions", total)
    col2.metric("Correct", correct)
    col3.metric("Accuracy", f"{(correct/total*100):.0f}%" if total else "—")
    col4.metric("Difficulty", sess.difficulty_level.title())

    st.divider()

    # ── Current topic ──
    current_topic = sess.current_topic or tracker.get_next_topic(sess)
    sess.current_topic = current_topic

    st.markdown(f"### 🎯 Current Topic: `{current_topic}`")

    topic_state = sess.topics.get(current_topic)
    if topic_state and topic_state.attempts > 0:
        score_pct = topic_state.score * 100
        st.progress(topic_state.score, text=f"Mastery: {score_pct:.0f}%")

    # Skip button
    skip_col, _ = st.columns([1, 5])
    with skip_col:
        if st.button("⏭️ I know this, skip"):
            sess = tracker.mark_topic_skipped(sess, current_topic)
            next_t = tracker.get_next_topic(sess)
            sess.current_topic = next_t
            st.session_state.current_session = sess
            st.session_state.qa_state = {}
            st.rerun()

    st.divider()

    # ── Phase: Generate Question ──
    if "question" not in qa:
        with st.spinner(f"🧠 Generating question on *{current_topic}*..."):
            state_in = {"session": sess}
            result = agents.assessment_agent(state_in)
            qa.update(result)
            st.session_state.qa_state = qa
            st.session_state.current_session = result["session"]
        st.rerun()

    # ── Phase: Show Question + Collect Answer ──
    question = qa.get("question", "")
    q_type = qa.get("question_type", "descriptive")

    st.markdown(f'<div class="question-box">❓ <strong>Question ({sess.difficulty_level.title()}):</strong><br><br>{question}</div>', unsafe_allow_html=True)

    if "feedback" not in qa:
        # Collect answer
        if q_type == "mcq":
            options = qa.get("mcq_options", {})
            if options:
                option_labels = [f"{k}: {v}" for k, v in options.items()]
                selected = st.radio("Choose your answer:", option_labels, key="mcq_choice")
                user_answer = selected.split(":")[0].strip() if selected else ""
            else:
                user_answer = st.text_input("Your answer:", key="desc_answer")
        else:
            user_answer = st.text_area("Your answer:", height=120, key="desc_answer", placeholder="Type your answer here...")

        col_submit, col_dontknow = st.columns([1, 1])
        with col_submit:
            submitted = st.button("✅ Submit Answer", type="primary")
        with col_dontknow:
            dont_know = st.button("🤷 I don't know — show answer")

        if submitted and user_answer:
            qa["user_answer"] = user_answer
            with st.spinner("Evaluating your answer..."):
                feedback_result = agents.explainer_agent(qa)
                qa.update(feedback_result)

            with st.spinner("Updating progress..."):
                progress_result = agents.progress_agent(qa)
                qa.update(progress_result)
                st.session_state.current_session = progress_result["session"]

            st.session_state.qa_state = qa
            st.rerun()

        elif dont_know:
            qa["user_answer"] = "I don't know"
            with st.spinner("Getting the answer..."):
                feedback_result = agents.explainer_agent(qa)
                qa.update(feedback_result)
            qa["is_correct"] = False
            with st.spinner("Updating progress..."):
                progress_result = agents.progress_agent(qa)
                qa.update(progress_result)
                st.session_state.current_session = progress_result["session"]
            st.session_state.qa_state = qa
            st.rerun()

    else:
        # ── Phase: Show Feedback ──
        correctness = qa.get("correctness_label", "")
        feedback = qa.get("feedback", "")

        if "Correct" == correctness:
            css_class = "feedback-correct"
            icon = "✅"
        elif "Partially" in correctness:
            css_class = "feedback-partial"
            icon = "🟡"
        else:
            css_class = "feedback-wrong"
            icon = "❌"

        st.markdown(
            f'<div class="{css_class}"><strong>{icon} {correctness}</strong><br><br>{feedback.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

        # ── Practice Question ──
        if "practice_question" not in qa:
            with st.spinner("Generating reinforcement question..."):
                pq_result = agents.quiz_agent(qa)
                qa.update(pq_result)
                st.session_state.qa_state = qa

        pq = qa.get("practice_question", "")
        pq_options = qa.get("practice_options", {})
        pq_correct = qa.get("practice_correct", "")
        pq_expl = qa.get("practice_explanation", "")

        if pq:
            with st.expander("🧪 Quick Practice Question", expanded=True):
                st.markdown(f"**{pq}**")
                if pq_options:
                    for k, v in pq_options.items():
                        icon_pq = "✅" if k == pq_correct else "  "
                        st.markdown(f"{icon_pq} **{k}:** {v}")
                    if pq_expl:
                        st.caption(f"Explanation: {pq_expl}")

        st.divider()
        if st.button("➡️ Next Question", type="primary"):
            st.session_state.qa_state = {}
            st.rerun()

    # ── Sidebar topic progress ──
    with st.sidebar:
        st.divider()
        st.subheader("📋 Topic Progress")
        sorted_topics = sorted(
            sess.topics.values(),
            key=lambda t: t.score,
        )
        for ts in sorted_topics[:10]:
            bar_val = ts.score
            label_color = "🔴" if ts.score < 0.4 else ("🟡" if ts.score < 0.75 else "🟢")
            attempts_str = f"({ts.attempts} tries)" if ts.attempts else "(not started)"
            st.markdown(f"{label_color} **{ts.topic}** {attempts_str}")
            if ts.attempts:
                st.progress(bar_val)

        if st.button("🏁 End Session & Summary"):
            st.session_state.page_nav = "📊 Dashboard"
            st.rerun()

# ──────────────────────────────────────────────
# Page: Dashboard
# ──────────────────────────────────────────────

def page_dashboard():
    st.header("📊 Dashboard")

    tracker = get_tracker()
    sess: SessionState = st.session_state.get("current_session")

    if sess:
        agents = get_agents()
        st.subheader(f"Session: {sess.syllabus_name}")
        summary = agents.get_session_summary(sess)
        st.markdown(summary)

        st.divider()

        # Topic mastery table
        st.subheader("Topic Mastery")
        topic_data = []
        for ts in sorted(sess.topics.values(), key=lambda t: t.score):
            status = "🔴 Weak" if ts.score < 0.4 else ("🟡 Developing" if ts.score < 0.75 else "🟢 Strong")
            topic_data.append({
                "Topic": ts.topic,
                "Mastery": f"{ts.score*100:.0f}%",
                "Attempts": ts.attempts,
                "Accuracy": f"{ts.accuracy*100:.0f}%" if ts.attempts else "—",
                "Status": status,
                "Skipped": "✓" if ts.skipped else "",
            })
        st.dataframe(topic_data, use_container_width=True)

        # Study recommendations
        weak = tracker.get_weak_topics(sess, n=5)
        if weak:
            st.divider()
            st.subheader("📌 Study Recommendations")
            st.markdown("Focus on these topics before your exam:")
            for t in weak:
                st.markdown(f"- **{t}**")

    else:
        st.info("No active session. Upload a syllabus to get started.")

    # All sessions list
    st.divider()
    st.subheader("All Sessions")
    sessions = tracker.list_sessions()
    if sessions:
        for sid, sname, updated in sessions:
            col1, col2, col3 = st.columns([3, 2, 1])
            col1.markdown(f"**{sname}**")
            col2.caption(f"Updated: {updated[:16]}")
            if col3.button("Load", key=f"load_{sid}"):
                loaded = tracker.load_session(sid)
                if loaded:
                    st.session_state.current_session = loaded
                    st.session_state.qa_state = {}
                    # Load RAG index if available
                    rag = get_rag()
                    if rag.has_index(sid):
                        rag.load_index(sid)
                    st.rerun()
    else:
        st.caption("No sessions yet.")

# ──────────────────────────────────────────────
# Main router
# ──────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "📤 Upload Syllabus":
        page_upload()
    elif page == "📚 Study Session":
        page_study()
    elif page == "📊 Dashboard":
        page_dashboard()


if __name__ == "__main__":
    main()
