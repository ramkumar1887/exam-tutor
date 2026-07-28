import os, re, time, pickle, logging
from pathlib import Path
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 50
TOP_K = 4

class RAGPipeline:
    def __init__(self, index_dir: str = "data/uploads"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: List[str] = []
        self._vectorizer = None
        self._matrix = None

    def ingest_pdf(self, pdf_path: str) -> List[str]:
        import fitz
        logger.info(f"[RAG] Ingesting PDF...")
        t0 = time.time()
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
        logger.info(f"[RAG] Extracted in {time.time()-t0:.2f}s")
        self._chunks = self._chunk_text(full_text)
        logger.info(f"[RAG] {len(self._chunks)} chunks created")
        return self._chunks

    def ingest_text(self, text: str, source: str = "") -> List[str]:
        self._chunks = self._chunk_text(text)
        return self._chunks

    def _chunk_text(self, text: str) -> List[str]:
        text = re.sub(r"\n{3,}", "\n\n", text)
        chunks, start = [], 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            if len(chunk) > 50:
                chunks.append(chunk)
            start = end - CHUNK_OVERLAP
        return chunks

    def build_index(self, session_id: str):
        if not self._chunks:
            return
        logger.info(f"[RAG] Building TF-IDF index on {len(self._chunks)} chunks...")
        t0 = time.time()
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._chunks)
        logger.info(f"[RAG] Index built in {time.time()-t0:.2f}s")  # will be < 1s
        # Save
        with open(self.index_dir / f"{session_id}.tfidf.pkl", "wb") as f:
            pickle.dump((self._chunks, self._vectorizer, self._matrix), f)

    def load_index(self, session_id: str) -> bool:
        path = self.index_dir / f"{session_id}.tfidf.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            self._chunks, self._vectorizer, self._matrix = pickle.load(f)
        return True

    def has_index(self, session_id: str) -> bool:
        return (self.index_dir / f"{session_id}.tfidf.pkl").exists()

    def retrieve(self, query: str, k: int = TOP_K) -> List[str]:
        if self._vectorizer is None or not self._chunks:
            return []
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_k = scores.argsort()[-k:][::-1]
        return [self._chunks[i] for i in top_k]

    def get_context_for_topic(self, topic: str, k: int = TOP_K) -> str:
        return "\n\n---\n\n".join(self.retrieve(topic, k=k))

    def extract_topics_with_llm(self, llm_client, raw_text: str) -> List[str]:
        text_snippet = raw_text[:3000]
        system = (
            "You are an expert at parsing academic syllabi. "
            "Extract all distinct exam topics as a JSON array of strings. "
            "Each topic should be 2-6 words, specific, and testable. "
            "Return only JSON, no markdown."
        )
        result = llm_client.call_json(system, f"Extract topics:\n\n{text_snippet}", max_tokens=400)
        if isinstance(result, list):
            topics = result
        elif isinstance(result, dict):
            topics = result.get("topics", result.get("items", []))
        else:
            topics = self._extract_topics_regex(raw_text)
        seen, clean = set(), []
        for t in topics:
            t = str(t).strip()
            if t and t.lower() not in seen and len(t) > 3:
                seen.add(t.lower())
                clean.append(t)
        return clean[:50]

    def _extract_topics_regex(self, text: str) -> List[str]:
        patterns = [r"^\s*\d+[\.\)]\s+(.+)$", r"^\s*[•\-\*]\s+(.+)$"]
        topics = []
        for line in text.split("\n"):
            for pat in patterns:
                m = re.match(pat, line.strip(), re.IGNORECASE)
                if m and 5 < len(m.group(1).strip()) < 100:
                    topics.append(m.group(1).strip())
                    break
        return topics