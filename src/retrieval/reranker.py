from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re
import httpx

from src.config import GENERATION_MODEL, OLLAMA_BASE_URL, RERANK_TOP_K
from src.retrieval.dense import RetrievalResult

RERANK_PROMPT = """Rate the relevance of this text chunk to the question on a scale of 0-10.
Only respond with a single integer number, nothing else.

Question: {question}

Text chunk:
{chunk}

Relevance score (0-10):"""

_client = httpx.Client(timeout=30.0, limits=httpx.Limits(max_keepalive_connections=10, max_connections=20))


class Reranker:
    def __init__(
        self,
        model: str = GENERATION_MODEL,
        top_k: int = RERANK_TOP_K,
        max_workers: int = 6,
    ):
        self.model = model
        self.top_k = top_k
        self.max_workers = max_workers

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        """Score candidates concurrently via parallel LLM-as-judge calls, keep top_k."""
        if not candidates:
            return []

        def _score_task(candidate: RetrievalResult) -> tuple[RetrievalResult, float]:
            score = self._score_relevance(query, candidate.content)
            return candidate, score

        workers = min(len(candidates), self.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            scored = list(executor.map(_score_task, candidates))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                chunk_id=c.chunk_id,
                content=c.content,
                score=relevance_score,
                metadata={**c.metadata, "rerank_score": relevance_score},
            )
            for c, relevance_score in scored[: self.top_k]
        ]

    def _score_relevance(self, question: str, chunk: str) -> float:
        prompt = RERANK_PROMPT.format(question=question, chunk=chunk[:1000])

        try:
            resp = _client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 5},
                },
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            match = re.search(r"\b(10|[0-9](\.[0-9]+)?)\b", text)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 10.0)
            return 0.0
        except (httpx.HTTPError, ValueError, KeyError):
            return 0.0
