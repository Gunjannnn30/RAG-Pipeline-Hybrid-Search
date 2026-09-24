from __future__ import annotations

import re

from src.config import SPARSE_TOP_K
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import VectorStore
from src.retrieval.dense import RetrievalResult


class SparseRetriever:
    def __init__(
        self,
        bm25_index: BM25Index | None = None,
        vector_store: VectorStore | None = None,
        top_k: int = SPARSE_TOP_K,
    ):
        self.bm25_index = bm25_index or BM25Index()
        self.vector_store = vector_store
        self.top_k = top_k
        self._chunk_content_cache: dict[str, str] = {}
        self._chunk_meta_cache: dict[str, dict] = {}

    def set_chunk_content(self, chunk_map: dict[str, str]) -> None:
        self._chunk_content_cache.update(chunk_map)

    def retrieve(self, query: str) -> list[RetrievalResult]:
        # Preprocess and clean query for BM25 while preserving identifiers and code tokens
        cleaned_query = self._preprocess_query(query)
        ranked = self.bm25_index.query(cleaned_query or query, top_k=self.top_k)

        # Optimization: batch-fetch all missing IDs in ONE ChromaDB call instead of N queries
        missing_ids = [
            chunk_id for chunk_id, _ in ranked
            if chunk_id not in self._chunk_content_cache and self.vector_store is not None
        ]
        if missing_ids and self.vector_store:
            try:
                got = self.vector_store.collection.get(
                    ids=missing_ids, include=["documents", "metadatas"]
                )
                if got and "ids" in got:
                    for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                        self._chunk_content_cache[cid] = doc
                        self._chunk_meta_cache[cid] = meta or {}
            except Exception:
                pass

        results = []
        for chunk_id, score in ranked:
            content = self._chunk_content_cache.get(chunk_id, "")
            metadata = {"retrieval_type": "sparse"}
            cached_meta = self._chunk_meta_cache.get(chunk_id)
            if cached_meta:
                metadata.update(cached_meta)

            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=content,
                score=score,
                metadata=metadata,
            ))

        return results

    def _preprocess_query(self, query: str) -> str:
        """Strip conversational filler and trailing question punctuation while preserving technical tokens."""
        q = re.sub(r"[?!.,;:\"']+$", "", query.strip())
        return q
