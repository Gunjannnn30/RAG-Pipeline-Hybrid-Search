# RAG with Hybrid Search Over Internal Docs ⚡

A high-performance, production-grade Retrieval-Augmented Generation (RAG) system that ingests documentation, indexes it with dual dense vector and sparse keyword search, retrieves the most relevant context using Reciprocal Rank Fusion (RRF), scores relevance with a parallel cross-encoder reranker, and generates grounded answers with inline verified citations — all powered 100% locally and offline via Ollama. No API keys. No cloud dependencies. Zero telemetry tracking.

Built and benchmarked on the [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench) dataset — 500K+ enterprise documents spanning Confluence, GitHub, Slack, Gmail, and Jira — with 500 golden Q&A pairs for rigorous evaluation.

---

## 🌟 What Makes This Different

Most RAG demos index a single PDF and stop. This system implements the production concerns that actually matter:

- **Parallel Hybrid Retrieval** — Concurrent dense semantic vector search (ChromaDB + Ollama) and sparse keyword search (BM25Okapi) executed in parallel threads, fused via Reciprocal Rank Fusion (RRF).
- **Parallel Cross-Encoder Reranking** — Multi-threaded concurrent LLM scoring of candidate chunks down to top-5 most relevant passages (~5x speedup).
- **Concurrent Citation Verification** — Every inline `[1]`, `[2]` citation claim is verified simultaneously in parallel against source chunks to catch hallucinations.
- **Query Embedding LRU Cache** — In-memory vector caching for 0ms lookup on repeated or quick-starter queries.
- **Batch High-Throughput Ingestion** — True batch embeddings via Ollama `/api/embed` with multi-threaded worker fallback.
- **Document Ingestion Studio** — Upload your own **PDF, Markdown, Plain Text, or HTML** documents through an interactive UI or REST API.
- **Context Window & Prompt Optimization** — Focuses generation on the top-5 chunks with `num_predict: 256`, completely preventing context window overflow and slow prompt evaluation timeouts.
- **Confidence Scoring & Abstention** — 3D scoring: retrieval relevance, citation coverage, and answer completeness. Low confidence triggers structured abstention instead of hallucination.
- **Three Chunking Strategies** — Fixed-size with overlap, recursive splitting by section headers, and semantic chunking by embedding similarity.
- **Near-Duplicate Detection** — Cosine similarity > 0.95 between chunk embeddings flags and skips duplicates, preventing context pollution.
- **Dual User Interfaces** — Full **Streamlit Enterprise Studio** (`:8501`) and **Zero-Install React 18 SPA** (`:8000/app`) with skeleton loaders and toast notifications.
- **59 Automated Tests** — Comprehensive unit, schema, and integration tests passing with 100% success rate.

---

## 🏗️ Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                    User Query                          │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │                                                   │
                    ▼                                                   ▼
     ┌─────────────────────────────┐                     ┌─────────────────────────────┐
     │     Dense Vector Search     │                     │     Sparse Keyword Search   │
     │  (Ollama nomic-embed-text)  │                     │          (BM25Okapi)        │
     │         + ChromaDB          │                     │     + Query Normalizer      │
     └──────────────┬──────────────┘                     └──────────────┬──────────────┘
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ Reciprocal Rank Fusion (RRF)    │
                             │ RRF(d) = w_d/(60+r_d) + w_s/... │
                             └────────────────┬────────────────┘
                                              │ Top 20 Candidates
                                              ▼
                             ┌─────────────────────────────────┐
                             │ Parallel Cross-Encoder Reranker │
                             │ (Concurrent LLM Judges, 0-10)   │
                             └────────────────┬────────────────┘
                                              │ Top 5 Focused Chunks
                                              ▼
                             ┌─────────────────────────────────┐
                             │   Grounded Answer Generator     │
                             │ (Ollama llama3.2:1b / llama3)   │
                             └────────────────┬────────────────┘
                                              │ Answer with [n] Citations
                                              ▼
                             ┌─────────────────────────────────┐
                             │ Concurrent Citation Verifier    │
                             │ (Parallel Claim Support Checks) │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ 3D Confidence Scorer & Guard    │
                             │ (Retrieval + Citation + Answer) │
                             └────────────────┬────────────────┘
                                              │
                         ┌────────────────────┴────────────────────┐
                         │                                         │
                         ▼                                         ▼
            [Score >= 0.3 Threshold]                   [Score < 0.3 Threshold]
            Grounded Answer + Citations                Structured Abstention
```

---

## 🖥️ User Interfaces

The platform offers two modern dark-themed web interfaces:

### 1. Streamlit Studio (`http://localhost:8501`)
- **Obsidian Slate Theme**: High-contrast, clean enterprise styling.
- **Telemetry Ribbon**: Live status of Ollama, ChromaDB, BM25, and indexed chunk count.
- **3D Confidence Telemetry**: Metric cards with color-coded gradient progress gauges.
- **Citation Drawer**: Supported vs. unsupported claim badges and source chunk inspection.
- **Document Ingestion Studio**: Drag-and-drop file upload with visual chunking strategy cards.
- **A/B Benchmark Arena**: Side-by-side comparison between Hybrid and Dense retrieval.

### 2. Zero-Install React 18 SPA (`http://localhost:8000/app`)
- **Modern React 18 & Tailwind CSS**: Served directly by FastAPI with zero Node.js/npm dependencies.
- **Zero Layout Shift**: Pulsing skeleton loaders maintain layout dimensions while generating answers.
- **Snackbar Toasts**: Non-intrusive floating notifications.

---

## ⚡ Performance Upgrades & Drawbacks Solved

| Bottleneck in Standard RAG | Our Solution | Result |
| :--- | :--- | :--- |
| **Sequential Reranking (40-50s)** | `ThreadPoolExecutor` parallel scoring with pooled HTTP | **~5x faster reranking** |
| **Sequential Citation Checks (10-15s)** | Concurrent multi-threaded claim verification | **~5x faster verification** |
| **Sequential Retrieval** | Parallel execution of dense and sparse search | **50% retrieval latency reduction** |
| **Redundant Query Embeddings** | In-memory `@lru_cache(maxsize=2048)` on query vectors | **0ms instantaneous repeat queries** |
| **Slow Document Indexing** | Batched embedding requests via `/api/embed` | **High-throughput document indexing** |
| **Context Window Overflows** | Capped generation context to top-5 chunks with `num_predict: 256` | **Zero generation timeouts** |

---

## 🚀 Quick Start (Local Setup)

### Prerequisites
- **Python 3.11+**
- **Ollama** ([ollama.com/download](https://ollama.com/download))
- **Git**

### 1. Clone & Install
```bash
git clone https://github.com/ujjwalredd/RAG-Pipeline.git
cd RAG-Pipeline

pip install -e ".[dev]"
pip install streamlit pymupdf python-multipart
```

### 2. Start Ollama and Pull Local Models
```bash
# Start Ollama engine (keep open)
ollama serve

# In another terminal, pull the required models
ollama pull nomic-embed-text    # Fast 768-dim embeddings (~274 MB)
ollama pull llama3.2:1b         # High-speed local LLM (~1.3 GB)
# Or full 8B model: ollama pull llama3:8b
```

### 3. Launch with Windows 1-Click Scripts
If you are on Windows, simply double-click:
- **`start_api.bat`** — Starts FastAPI on `http://localhost:8000` (and React SPA on `/app`)
- **`start_dashboard.bat`** — Starts Streamlit Studio on `http://localhost:8501`
- **`share_online.bat`** — Creates an instant public HTTPS tunnel
- **`run_tests.bat`** — Runs all 59 automated tests
- **`run_eval.bat`** — Runs the golden evaluation benchmark

---

## 🌐 Deploying Online

For complete, step-by-step instructions on deploying the pipeline online, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

### Quick Options:
1. **Instant 60-Second Free Tunnel (Zero Server Setup)**:
   ```bash
   cloudflared tunnel --url http://localhost:8501
   ```
   Generates an immediate public HTTPS link (e.g. `https://xxx.trycloudflare.com`) accessible from any device.

2. **24/7 Production Cloud VPS with Docker Compose**:
   ```bash
   docker compose up -d --build
   ```
   Orchestrates Ollama, FastAPI, and Streamlit with automatic model pulling, volume persistence, and restart policies.

---

## 📡 REST API Reference

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### `POST /v1/ask` — Query with Grounded Citations
```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the deployment guidelines and rate limits?",
    "retrieval_mode": "hybrid",
    "use_reranker": false,
    "dense_weight": 0.7,
    "sparse_weight": 0.3
  }'
```

### `POST /v1/upload` — Ingest User Documents
```bash
curl -X POST http://localhost:8000/v1/upload \
  -F "files=@my_document.pdf" \
  -F "chunking_strategy=recursive"
```

### `GET /v1/documents` — List Indexed Documents
```bash
curl http://localhost:8000/v1/documents
```

### `GET /health` — Cluster Health Status
```bash
curl http://localhost:8000/health
```

---

## 🧪 Testing & Evaluation

### Run Automated Unit Tests (59/59 passing)
```bash
pytest tests/ -v
```

### Run Golden Benchmark Evaluation
```bash
python -m src.evaluation.run_eval
```

---

## 📂 Project Structure

```
├── DEPLOYMENT.md              # Cloud VPS, Docker Compose, and tunnel guide
├── Dockerfile                 # Production multi-stage container
├── docker-compose.yml         # Turn-key multi-container orchestration
├── pyproject.toml             # Python package configuration
├── run_eval.bat               # 1-click evaluation benchmark launcher
├── run_tests.bat              # 1-click test suite runner
├── share_online.bat           # 1-click Cloudflare public sharing tunnel
├── start_api.bat              # 1-click FastAPI backend launcher
├── start_dashboard.bat        # 1-click Streamlit Studio launcher
├── src/
│   ├── api/
│   │   ├── main.py            # FastAPI endpoints (/v1/ask, /v1/upload, /app)
│   │   ├── schemas.py         # Pydantic request & response schemas
│   │   └── static/index.html  # Zero-install React 18 + Tailwind SPA
│   ├── chunking/              # Recursive, fixed-size, and semantic chunking
│   ├── config.py              # Centralized configuration & model defaults
│   ├── dashboard/app.py       # Streamlit Studio with Obsidian Slate theme
│   ├── evaluation/            # Automated metrics & benchmark runners
│   ├── generation/            # Grounded generator, parallel citations, confidence
│   ├── indexing/              # Vector store, BM25, embeddings, deduplication
│   ├── ingestion/             # Multi-format document loader (PDF, MD, HTML, TXT)
│   └── retrieval/             # Parallel hybrid search, RRF fusion, reranker
└── tests/                     # 59 automated test cases
```

---

## 📄 License

MIT License. Free for commercial and personal use.
