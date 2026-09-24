from __future__ import annotations

import httpx

from src.config import GENERATION_MODEL, OLLAMA_BASE_URL
from src.generation.prompts import CONTEXT_TEMPLATE, SYSTEM_PROMPT
from src.retrieval.dense import RetrievalResult

_client = httpx.Client(
    timeout=180.0,
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
)


class GroundedGenerator:
    def __init__(self, model: str = GENERATION_MODEL):
        self.model = model

    def generate(self, question: str, context_chunks: list[RetrievalResult]) -> str:
        context_blocks = self._format_context(context_chunks)
        prompt = CONTEXT_TEMPLATE.format(
            context_blocks=context_blocks,
            question=question,
        )

        resp = _client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": self.model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 256},
            },
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def _format_context(self, chunks: list[RetrievalResult]) -> str:
        blocks = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.metadata.get("title", "Unknown")
            source = chunk.metadata.get("source_type", "unknown")
            blocks.append(f"[{i}] (Source: {title} | Type: {source})\n{chunk.content}")
        return "\n\n---\n\n".join(blocks)
