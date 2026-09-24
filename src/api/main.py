from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.dependencies import get_app_state
from src.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkResponse,
    CitationResponse,
    ConfidenceResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from src.config import OLLAMA_BASE_URL
from src.indexing.pipeline import IndexingPipeline
from src.ingestion.enterprise_loader import load_enterprise_dataset
from src.ingestion.loader import load_uploaded_file


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = get_app_state()
    state._load_bm25()
    yield


app = FastAPI(
    title="RAG with Hybrid Search API",
    description="High-performance hybrid dense+BM25 search, parallel reranking, and citation verification",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/app", response_class=FileResponse)
    def serve_react_app():
        return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health", response_model=HealthResponse)
def health_check():
    state = get_app_state()
    ollama_ok = False
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        ollama_ok = resp.status_code == 200
    except httpx.HTTPError:
        pass

    return HealthResponse(
        status="ok",
        indexed_chunks=state.vector_store.count(),
        ollama_available=ollama_ok,
    )


@app.post("/v1/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    state = get_app_state()

    if state.vector_store.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. POST /v1/ingest first.")

    pipeline = state.pipeline

    pipeline.retriever.dense_weight = request.dense_weight
    pipeline.retriever.sparse_weight = request.sparse_weight

    try:
        if request.retrieval_mode == "hybrid":
            result = pipeline.ask(request.question, use_reranker=request.use_reranker)
            return _format_response(result)

        if request.retrieval_mode == "dense":
            chunks = pipeline.retriever.retrieve_dense_only(request.question)
        else:
            chunks = pipeline.retriever.retrieve_sparse_only(request.question)

        answer = pipeline.generator.generate(request.question, chunks)
        citations = pipeline.verifier.verify(answer, chunks)
        confidence = pipeline.scorer.score(request.question, answer, chunks, citations)

        from src.generation.pipeline import GenerationResult
        result = GenerationResult(
            question=request.question,
            answer=answer,
            context_chunks=chunks,
            citations=citations,
            confidence=confidence,
            is_confident=confidence.composite >= pipeline.confidence_threshold,
        )
        return _format_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/documents", response_model=DocumentListResponse)
def list_documents():
    state = get_app_state()
    collection = state.vector_store.collection

    result = collection.get(include=["metadatas"])
    if not result["metadatas"]:
        return DocumentListResponse(total=0, documents=[])

    seen: dict[str, DocumentInfo] = {}
    for meta in result["metadatas"]:
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in seen:
            seen[doc_id] = DocumentInfo(
                doc_id=doc_id,
                title=meta.get("title", ""),
                source_type=meta.get("source_type", ""),
                char_count=meta.get("char_count", 0),
            )
        else:
            seen[doc_id].char_count += meta.get("char_count", 0)

    docs = list(seen.values())
    return DocumentListResponse(total=len(docs), documents=docs)


@app.post("/v1/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest):
    documents = load_enterprise_dataset(
        max_docs=request.max_docs,
        source_types=request.source_types,
    )

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found matching criteria.")

    indexer = IndexingPipeline(strategy=request.chunking_strategy)
    stats = indexer.run(documents)

    state = get_app_state()
    state.reload()

    return IngestResponse(**stats)


@app.post("/v1/upload", response_model=IngestResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    chunking_strategy: str = Form("recursive"),
):
    """Upload custom files (PDF, Markdown, Text, HTML) and index them."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    valid_strategies = ["recursive", "fixed_size", "semantic"]
    if chunking_strategy not in valid_strategies:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid chunking strategy '{chunking_strategy}'. Must be one of {valid_strategies}",
        )

    documents = []
    for file in files:
        content_bytes = await file.read()
        if not content_bytes:
            continue
        try:
            doc = load_uploaded_file(file.filename, content_bytes)
            documents.append(doc)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process '{file.filename}': {e}")

    if not documents:
        raise HTTPException(status_code=400, detail="No valid document content could be extracted.")

    indexer = IndexingPipeline(strategy=chunking_strategy)
    stats = indexer.run(documents)

    state = get_app_state()
    state.reload()

    return IngestResponse(**stats)


@app.post("/v1/clear")
def clear_knowledge_base():
    """Reset both ChromaDB vector store and BM25 index."""
    state = get_app_state()
    state.vector_store.reset()
    state.bm25_index.clear()
    state.reload()
    return {
        "status": "ok",
        "message": "Knowledge base reset successfully.",
        "indexed_chunks": state.vector_store.count(),
    }


def _format_response(result) -> AskResponse:
    return AskResponse(
        question=result.question,
        answer=result.answer,
        is_confident=result.is_confident,
        confidence=ConfidenceResponse(
            retrieval_confidence=result.confidence.retrieval_confidence,
            citation_coverage=result.confidence.citation_coverage,
            answer_completeness=result.confidence.answer_completeness,
            composite=result.confidence.composite,
        ),
        citations=[
            CitationResponse(
                citation_id=c.citation_id,
                claim=c.claim,
                source_chunk_id=c.source_chunk_id,
                supported=c.supported,
            )
            for c in result.citations
        ],
        context_chunks=[
            ChunkResponse(
                chunk_id=c.chunk_id,
                content=c.content,
                score=c.score,
                metadata=c.metadata,
            )
            for c in result.context_chunks
        ],
    )
