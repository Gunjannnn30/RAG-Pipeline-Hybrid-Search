from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import httpx

from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL

_client = httpx.Client(
    timeout=120.0,
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
)


def embed_texts(
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed a list of texts via Ollama API using batching with concurrent fallback."""
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = _client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": model, "input": batch},
            )
            if resp.status_code == 200:
                data = resp.json()
                if "embeddings" in data and len(data["embeddings"]) == len(batch):
                    all_embeddings.extend(data["embeddings"])
                    continue
        except httpx.HTTPError:
            pass

        # Fallback: concurrent embedding workers
        workers = min(len(batch), 6)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            batch_embeddings = list(executor.map(lambda t: _embed_one(t, model), batch))
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def _embed_one(text: str, model: str) -> list[float]:
    # Try new endpoint first (/api/embed), fall back to legacy (/api/embeddings)
    try:
        resp = _client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": text},
        )
        if resp.status_code == 200:
            data = resp.json()
            if "embeddings" in data and data["embeddings"]:
                return data["embeddings"][0]
            if "embedding" in data:
                return data["embedding"]
    except httpx.HTTPError:
        pass

    # Legacy endpoint
    resp = _client.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": model, "prompt": text},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


@lru_cache(maxsize=2048)
def _embed_single_cached(text: str, model: str) -> tuple[float, ...]:
    return tuple(_embed_one(text, model))


def embed_single(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Embed a single string with an in-memory LRU cache for 0ms repeated query latency."""
    return list(_embed_single_cached(text, model))
