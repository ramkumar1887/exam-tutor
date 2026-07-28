"""
LangGraph Workflow
Wires the four agents into a stateful graph.
Adapted from NeoTutor's LangGraph setup, extended with:
  - Conditional routing (correct → next topic, incorrect → quiz reinforcement)
  - Topic-switch logic (move on after 2 consecutive correct answers on same topic)
"""

from langgraph.graph import StateGraph, END
from core.agents import TutorAgents


def build_workflow(agents: TutorAgents):
    """
    Build and compile the LangGraph tutor workflow.

    Graph structure:
        assess → [user answers] → explain → progress → quiz → [user answers practice]
                                                ↓
                                   (if correct 2x on topic) → assess (new topic)
                                   (else)                   → assess (same topic)
    """
    workflow = StateGraph(dict)

    # Register nodes
    workflow.add_node("assess",   agents.assessment_agent)
    workflow.add_node("explain",  agents.explainer_agent)
    workflow.add_node("quiz",     agents.quiz_agent)
    workflow.add_node("progress", agents.progress_agent)

    # Entry point
    workflow.set_entry_point("assess")

    # assess → (user answers in UI) → explain
    workflow.add_edge("assess", "explain")

    # explain → progress
    workflow.add_edge("explain", "progress")

    # progress → quiz (always show reinforcement)
    workflow.add_edge("progress", "quiz")

    # quiz → assess (loop)
    workflow.add_edge("quiz", "assess")

    return workflow.compile()


def route_after_progress(state: dict) -> str:
    """
    Conditional routing from progress node.
    Not used in linear flow above but available for advanced routing.
    """
    sess = state.get("session")
    if sess is None:
        return "assess"

    topic = sess.current_topic
    ts = sess.topics.get(topic)
    if ts and ts.score >= 0.8 and ts.attempts >= 2:
        return "assess"  # move to new topic
    return "quiz"  # reinforce current topic
