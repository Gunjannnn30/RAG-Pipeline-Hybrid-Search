from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import re

import httpx

from src.config import GENERATION_MODEL, OLLAMA_BASE_URL
from src.generation.prompts import CITATION_VERIFY_PROMPT
from src.retrieval.dense import RetrievalResult

_client = httpx.Client(timeout=30.0, limits=httpx.Limits(max_keepalive_connections=10, max_connections=20))


@dataclass
class CitationResult:
    citation_id: int
    claim: str
    source_chunk_id: str
    supported: bool


class CitationVerifier:
    def __init__(self, model: str = GENERATION_MODEL, max_workers: int = 6):
        self.model = model
        self.max_workers = max_workers

    def verify(
        self, answer: str, context_chunks: list[RetrievalResult]
    ) -> list[CitationResult]:
        claims = self._extract_claims_with_citations(answer)
        if not claims:
            return []

        def _verify_claim(item: tuple[int, str]) -> CitationResult:
            citation_id, claim = item
            if citation_id < 1 or citation_id > len(context_chunks):
                return CitationResult(
                    citation_id=citation_id,
                    claim=claim,
                    source_chunk_id="INVALID",
                    supported=False,
                )

            chunk = context_chunks[citation_id - 1]
            supported = self._check_support(claim, chunk.content)
            return CitationResult(
                citation_id=citation_id,
                claim=claim,
                source_chunk_id=chunk.chunk_id,
                supported=supported,
            )

        workers = min(len(claims), self.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_verify_claim, claims))

        return results

    def _extract_claims_with_citations(self, answer: str) -> list[tuple[int, str]]:
        """Extract (citation_number, sentence_containing_citation) pairs."""
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        claims = []

        for sentence in sentences:
            refs = re.findall(r"\[(\d+)\]", sentence)
            clean_sentence = re.sub(r"\s*\[\d+\]", "", sentence).strip()
            if not clean_sentence:
                continue
            for ref in refs:
                claims.append((int(ref), clean_sentence))

        return claims

    def _check_support(self, claim: str, source_text: str) -> bool:
        prompt = CITATION_VERIFY_PROMPT.format(
            claim=claim,
            source=source_text[:1500],
        )

        try:
            resp = _client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 10},
                },
            )
            resp.raise_for_status()
            verdict = resp.json().get("response", "").strip().upper()
            return "SUPPORTED" in verdict
        except (httpx.HTTPError, KeyError):
            return False
