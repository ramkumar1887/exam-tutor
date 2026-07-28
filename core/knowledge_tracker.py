"""
Knowledge State Tracker
Tracks per-topic mastery scores (0.0–1.0) across sessions using SQLite.
Decides which topics to focus on next (weakest first).
"""

import sqlite3
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("TUTOR_DB_PATH", "data/sessions/knowledge.db")


@dataclass
class TopicState:
    topic: str
    score: float = 0.0          # 0.0 = unknown, 1.0 = mastered
    attempts: int = 0
    correct: int = 0
    last_seen: Optional[str] = None
    skipped: bool = False        # User explicitly said "I know this"

    @property
    def accuracy(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.correct / self.attempts

    @property
    def priority(self) -> float:
        """Lower = needs more study. Used to sort topics."""
        if self.skipped:
            return 1.0  # Treat as known
        # Weight: low score, high attempts without success → high priority
        return self.score


@dataclass
class SessionState:
    session_id: str
    syllabus_name: str
    topics: Dict[str, TopicState] = field(default_factory=dict)
    current_topic: Optional[str] = None
    question_history: List[str] = field(default_factory=list)
    total_questions: int = 0
    total_correct: int = 0
    difficulty_level: str = "medium"
    performance_window: List[bool] = field(default_factory=list)  # last N answers
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    extra: Dict = field(default_factory=dict)  # arbitrary extra state for agents


class KnowledgeTracker:
    """
    Manages per-user, per-syllabus topic mastery.
    Persists to SQLite so state survives across Streamlit reruns.
    """

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    syllabus_name TEXT,
                    state_json TEXT,
                    updated_at TEXT
                )
            """)
            conn.commit()

    # ──────────────────────────────────────────────
    # Session CRUD
    # ──────────────────────────────────────────────

    def create_session(self, session_id: str, syllabus_name: str, topics: List[str]) -> SessionState:
        state = SessionState(
            session_id=session_id,
            syllabus_name=syllabus_name,
            topics={t: TopicState(topic=t) for t in topics},
        )
        self._save(state)
        return state

    def load_session(self, session_id: str) -> Optional[SessionState]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT state_json FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        if not row:
            return None
        return self._deserialize(row[0])

    def save_session(self, state: SessionState):
        self._save(state)

    def list_sessions(self) -> List[Tuple[str, str, str]]:
        """Returns list of (session_id, syllabus_name, updated_at)"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT session_id, syllabus_name, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return rows

    # ──────────────────────────────────────────────
    # Topic selection logic
    # ──────────────────────────────────────────────

    def get_next_topic(self, state: SessionState) -> str:
        """
        Returns the topic that needs the most attention:
        - Not skipped
        - Lowest score
        - Tie-break: least recently seen
        """
        candidates = [
            ts for ts in state.topics.values()
            if not ts.skipped
        ]
        if not candidates:
            # All skipped — reset skips and start over
            for ts in state.topics.values():
                ts.skipped = False
            candidates = list(state.topics.values())

        # Sort by (score ASC, last_seen ASC)
        def sort_key(ts: TopicState):
            last = ts.last_seen or "0000"
            return (ts.priority, last)

        candidates.sort(key=sort_key)
        return candidates[0].topic

    def get_weak_topics(self, state: SessionState, n: int = 5) -> List[str]:
        """Return top-n weakest topics for the summary."""
        candidates = [ts for ts in state.topics.values() if not ts.skipped and ts.attempts > 0]
        candidates.sort(key=lambda ts: ts.score)
        return [ts.topic for ts in candidates[:n]]

    # ──────────────────────────────────────────────
    # Score update
    # ──────────────────────────────────────────────

    def record_answer(self, state: SessionState, topic: str, is_correct: bool) -> SessionState:
        """Update topic score after an answer."""
        if topic not in state.topics:
            state.topics[topic] = TopicState(topic=topic)

        ts = state.topics[topic]
        ts.attempts += 1
        ts.last_seen = datetime.now().isoformat()

        if is_correct:
            ts.correct += 1

        # Exponential moving average for score
        alpha = 0.3  # learning rate
        ts.score = (1 - alpha) * ts.score + alpha * (1.0 if is_correct else 0.0)

        # Session-level tracking
        state.total_questions += 1
        if is_correct:
            state.total_correct += 1
        state.performance_window.append(is_correct)
        if len(state.performance_window) > 5:
            state.performance_window = state.performance_window[-5:]

        # Adaptive difficulty
        if len(state.performance_window) >= 3:
            recent_acc = sum(state.performance_window[-3:]) / 3
            if recent_acc >= 0.8 and state.difficulty_level != "hard":
                state.difficulty_level = "hard" if state.difficulty_level == "medium" else "medium"
            elif recent_acc <= 0.33 and state.difficulty_level != "easy":
                state.difficulty_level = "easy" if state.difficulty_level == "medium" else "medium"

        self._save(state)
        return state

    def mark_topic_skipped(self, state: SessionState, topic: str) -> SessionState:
        if topic in state.topics:
            state.topics[topic].skipped = True
            state.topics[topic].score = 1.0
        self._save(state)
        return state

    # ──────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────

    def _save(self, state: SessionState):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions (session_id, syllabus_name, state_json, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (state.session_id, state.syllabus_name, self._serialize(state), datetime.now().isoformat()),
            )
            conn.commit()

    def _serialize(self, state: SessionState) -> str:
        d = asdict(state)
        return json.dumps(d)

    def _deserialize(self, json_str: str) -> SessionState:
        d = json.loads(json_str)
        topics = {k: TopicState(**v) for k, v in d.pop("topics", {}).items()}
        state = SessionState(**d, topics=topics)
        return state
