from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.config import (
    DENSE_TOP_K,
    FUSION_TOP_K,
    RERANK_TOP_K,
    RRF_DENSE_WEIGHT,
    RRF_SPARSE_WEIGHT,
    SPARSE_TOP_K,
)
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import VectorStore
from src.retrieval.dense import DenseRetriever, RetrievalResult
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever


class HybridRetriever:
    """Full hybrid retrieval: concurrent dense + sparse → RRF fusion → parallel reranker."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
        dense_top_k: int = DENSE_TOP_K,
        sparse_top_k: int = SPARSE_TOP_K,
        fusion_top_k: int = FUSION_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        dense_weight: float = RRF_DENSE_WEIGHT,
        sparse_weight: float = RRF_SPARSE_WEIGHT,
        use_reranker: bool = True,
    ):
        self.dense = DenseRetriever(vector_store=vector_store, top_k=dense_top_k)
        self.sparse = SparseRetriever(
            bm25_index=bm25_index, vector_store=vector_store, top_k=sparse_top_k
        )
        self.reranker = Reranker(top_k=rerank_top_k) if use_reranker else None
        self.fusion_top_k = fusion_top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(
        self, query: str, use_reranker: bool | None = None
    ) -> list[RetrievalResult]:
        # Run dense vector search and sparse BM25 keyword search concurrently in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(self.dense.retrieve, query)
            future_sparse = executor.submit(self.sparse.retrieve, query)
            dense_results = future_dense.result()
            sparse_results = future_sparse.result()

        fused = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            top_k=self.fusion_top_k,
        )

        should_rerank = use_reranker if use_reranker is not None else (self.reranker is not None)
        if should_rerank and self.reranker:
            return self.reranker.rerank(query, fused)

        return fused

    def retrieve_dense_only(self, query: str) -> list[RetrievalResult]:
        return self.dense.retrieve(query)

    def retrieve_sparse_only(self, query: str) -> list[RetrievalResult]:
        return self.sparse.retrieve(query)
