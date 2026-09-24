import os
import time
import httpx
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="RAG with Hybrid Search · Production Retrieval Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Professional Enterprise CSS (uiux-designer & frontend-expert) ---
st.markdown("""
<style>
    /* Google Fonts & Root CSS Variables */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-canvas: #090d16;
        --bg-surface: #131b2e;
        --bg-card-hover: #17223b;
        --border-subtle: #202d45;
        --border-active: #3b82f6;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --brand-primary: #6366f1;
        --brand-cyan: #06b6d4;
        --status-success: #10b981;
        --status-warning: #f59e0b;
        --status-danger: #f43f5e;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    /* Top Ribbon Header */
    .telemetry-ribbon {
        background: linear-gradient(135deg, rgba(19, 27, 46, 0.95) 0%, rgba(9, 13, 22, 0.98) 100%);
        border: 1px solid #202d45;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 22px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .hero-title-group h1 {
        font-size: 24px;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .hero-tagline {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
        font-weight: 500;
    }

    .telemetry-badges {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        border: 1px solid transparent;
    }

    .badge-online {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.3);
    }

    .badge-offline {
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border-color: rgba(244, 63, 94, 0.3);
    }

    .badge-neutral {
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        border-color: rgba(99, 102, 241, 0.28);
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }

    .metric-card {
        background: #131b2e;
        border: 1px solid #202d45;
        border-radius: 12px;
        padding: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 800;
        margin-top: 4px;
        color: #f8fafc;
    }

    /* Grounded Answer Card */
    .answer-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 22px;
        margin: 18px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .answer-card.abstention {
        border-left-color: #f59e0b;
        background: rgba(245, 158, 11, 0.04);
    }

    .answer-text {
        font-size: 15.5px;
        line-height: 1.7;
        color: #f3f4f6;
    }

    /* Citations Visualizer */
    .citation-card {
        background: #131b2e;
        border: 1px solid #202d45;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: border-color 0.2s ease;
    }

    .citation-card.supported {
        border-left: 4px solid #10b981;
    }

    .citation-card.unsupported {
        border-left: 4px solid #f43f5e;
    }

    /* Strategy Card */
    .strategy-card {
        background: #131b2e;
        border: 1px solid #202d45;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
    }

    /* Skeleton Placeholder (Frontend Expert: prevent layout shift) */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .skeleton-box {
        animation: pulse 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        background: #1e293b;
        border-radius: 8px;
    }

    /* Button Styling Overrides */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 13.5px;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


# --- System Health Fetcher ---
def get_system_health():
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "error", "indexed_chunks": 0, "ollama_available": False}


health_data = get_system_health()
ollama_active = health_data.get("ollama_available", False)
chunk_count = health_data.get("indexed_chunks", 0)

# --- Top Telemetry Ribbon ---
ollama_badge_class = "badge-online" if ollama_active else "badge-offline"
ollama_status_text = "Ollama Active" if ollama_active else "Ollama Offline"

st.markdown(f"""
<div class="telemetry-ribbon">
    <div class="hero-title-group">
        <h1>⚡ RAG with Hybrid Search</h1>
        <div class="hero-tagline">
            Dual Dense Vector + BM25 Keyword Hybrid Search · Parallel Cross-Encoder Reranker · Factual Citation Verification · 100% Offline
        </div>
    </div>
    <div class="telemetry-badges">
        <span class="badge-pill {ollama_badge_class}">● {ollama_status_text}</span>
        <span class="badge-pill badge-neutral">⚡ ChromaDB Cosine</span>
        <span class="badge-pill badge-neutral">🔍 BM25Okapi</span>
        <span class="badge-pill badge-neutral">🚀 Parallel Engine</span>
        <span class="badge-pill badge-neutral">📚 {chunk_count:,} Chunks Indexed</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Sidebar Navigation & Tuners ---
with st.sidebar:
    st.markdown("### 🎛️ Retrieval Engine Tuning")
    
    retrieval_mode = st.radio(
        "Search Strategy",
        ["hybrid", "dense", "sparse"],
        format_func=lambda x: {
            "hybrid": "⚡ Hybrid (Vectors + BM25)",
            "dense": "🎯 Dense Vectors Only",
            "sparse": "🔍 Sparse BM25 Only",
        }[x],
        help="Hybrid applies Reciprocal Rank Fusion to merge semantic vectors and exact keyword matches."
    )

    use_reranker = st.toggle(
        "Cross-Encoder Reranker",
        value=False,
        help="Applies an LLM judge to score top-20 candidate passages down to top-5 for maximum precision."
    )

    if retrieval_mode == "hybrid":
        st.markdown("#### RRF Weight Distribution")
        st.caption("Reciprocal Rank Fusion equation:")
        st.latex(r"RRF(d) = w_d \cdot \frac{1}{60 + r_d} + w_s \cdot \frac{1}{60 + r_s}")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            dense_weight = st.slider("Dense (Vectors)", 0.0, 1.0, 0.7, 0.05)
        with col_w2:
            sparse_weight = st.slider("Sparse (BM25)", 0.0, 1.0, 0.3, 0.05)
    else:
        dense_weight, sparse_weight = 0.7, 0.3

    st.markdown("---")
    st.markdown("### 📡 Engine Telemetry")
    st.markdown(f"""
    - **API Base:** `{API_BASE}`
    - **Embeddings:** `nomic-embed-text (768d)`
    - **Inference LLM:** `llama3.2:1b (Ollama)`
    - **Knowledge Base:** `{chunk_count:,} chunks`
    """)

    st.markdown("---")
    with st.expander("🛡️ Maintenance & Reset", expanded=False):
        st.caption("Purge all documents from ChromaDB and the BM25 index.")
        if st.button("🗑️ Reset Knowledge Base", type="secondary", use_container_width=True):
            try:
                res = httpx.post(f"{API_BASE}/v1/clear", timeout=10.0)
                if res.status_code == 200:
                    st.toast("Knowledge base cleared successfully!", icon="✅")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.toast(f"Reset failed: {res.text}", icon="❌")
            except Exception as e:
                st.toast(f"Error resetting: {e}", icon="⚠️")


# --- Main Tab View ---
tab_chat, tab_upload, tab_compare, tab_docs = st.tabs([
    "💬 Search & Grounded Q&A",
    "📤 Document Ingestion Studio",
    "⚖️ Hybrid vs. Dense A/B Arena",
    "📚 Document Repository"
])


# =========================================================================
# TAB 1: Search & Grounded Q&A
# =========================================================================
with tab_chat:
    # Example Prompt Suggestions (Pill Cards)
    st.markdown("**💡 Quick Query Starters:**")
    col_p1, col_p2, col_p3 = st.columns(3)
    if col_p1.button("⚡ Deployment guidelines & rate limits?", use_container_width=True):
        st.session_state["query_input"] = "What are the deployment guidelines and rate limits?"
    if col_p2.button("📄 Confluence documentation rules?", use_container_width=True):
        st.session_state["query_input"] = "What information is described in Confluence documents?"
    if col_p3.button("❓ Out-of-scope test (chocolate cake)?", use_container_width=True):
        st.session_state["query_input"] = "What is the recipe for baking chocolate cake?"

    default_q = st.session_state.get("query_input", "")
    query = st.text_input(
        "Enter your query across indexed documents:",
        value=default_q,
        placeholder="e.g., What are the deployment guidelines and rate limits?",
        key="main_query_input"
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        submit = st.button("🔍 Search & Generate", type="primary", use_container_width=True)

    if submit and query:
        if chunk_count == 0:
            st.warning("⚠️ No documents indexed yet. Please upload files in the 'Document Ingestion Studio' tab.")
        else:
            with st.spinner("Executing retrieval fusion, LLM inference, and citation verification..."):
                start_time = time.time()
                try:
                    payload = {
                        "question": query,
                        "retrieval_mode": retrieval_mode,
                        "use_reranker": use_reranker,
                        "dense_weight": dense_weight,
                        "sparse_weight": sparse_weight,
                    }
                    resp = httpx.post(f"{API_BASE}/v1/ask", json=payload, timeout=120.0)
                    elapsed = time.time() - start_time

                    if resp.status_code == 200:
                        data = resp.json()
                        is_confident = data.get("is_confident", True)
                        answer = data.get("answer", "")
                        citations = data.get("citations", [])
                        chunks = data.get("context_chunks", [])
                        conf = data.get("confidence", {})

                        # Render Grounded Answer Card
                        abstention_class = "" if is_confident else "abstention"
                        badge_label = "Verified Grounded Answer" if is_confident else "Low-Confidence Abstention"
                        badge_color = "#10b981" if is_confident else "#f59e0b"

                        st.markdown(f"""
                        <div class="answer-card {abstention_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <span style="font-weight: 700; font-size: 13px; color: {badge_color}; text-transform: uppercase; letter-spacing: 0.05em;">
                                    ● {badge_label}
                                </span>
                                <span style="font-size: 12px; color: #94a3b8;">
                                    ⏱️ {elapsed:.2f}s · Mode: <code>{retrieval_mode}</code> · Reranker: <code>{use_reranker}</code>
                                </span>
                            </div>
                            <div class="answer-text">
                                {answer}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if not is_confident:
                            st.warning("⚠️ **Abstention Triggered**: Grounding confidence fell below the safety threshold. The model declined to extrapolate and provided partial factual findings above.")

                        # Confidence Gauge Metrics (3D Telemetry)
                        st.markdown("### 🎯 3D Confidence Score Breakdown")
                        comp_score = conf.get("composite", 0.0)
                        ret_score = conf.get("retrieval_confidence", 0.0)
                        cit_score = conf.get("citation_coverage", 0.0)
                        ans_score = conf.get("answer_completeness", 0.0)

                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        col_m1.metric("Composite Score", f"{comp_score:.1%}")
                        col_m2.metric("Retrieval Relevance", f"{ret_score:.1%}")
                        col_m3.metric("Citation Coverage", f"{cit_score:.1%}")
                        col_m4.metric("Answer Completeness", f"{ans_score:.1%}")

                        # Progress Bar Color Gauge
                        st.progress(min(max(comp_score, 0.0), 1.0))

                        # Citations Section
                        if citations:
                            st.markdown(f"### 🛡️ Verified Inline Citations ({len(citations)} references verified)")
                            for cit in citations:
                                supported = cit.get("supported", False)
                                status_text = "SUPPORTED CLAIM" if supported else "UNSUPPORTED CLAIM"
                                card_class = "supported" if supported else "unsupported"
                                status_color = "#10b981" if supported else "#f43f5e"

                                st.markdown(f"""
                                <div class="citation-card {card_class}">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                                        <b style="color: {status_color}; font-size: 13px;">[{cit.get('citation_id')}] {status_text}</b>
                                        <code style="font-size: 11px;">{cit.get('source_chunk_id')}</code>
                                    </div>
                                    <div style="font-size: 13.5px; color: #e2e8f0; margin-bottom: 4px;">
                                        <b>Claimed:</b> <i>"{cit.get('claim')}"</i>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                        # Retrieved Passages Explorer
                        if chunks:
                            with st.expander(f"📑 Retrieved Context Passages ({len(chunks)} chunks used)"):
                                for idx, ch in enumerate(chunks, 1):
                                    st.markdown(f"**[{idx}] {ch['metadata'].get('title', 'Document')}** — `Score: {ch['score']:.4f}` | `Source: {ch['metadata'].get('source_type', 'txt')}`")
                                    st.code(ch.get("content", ""), language="text")

                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Inference request failed: {e}")


# =========================================================================
# TAB 2: Document Ingestion Studio
# =========================================================================
with tab_upload:
    st.subheader("📤 Ingest Custom Knowledge into the Pipeline")
    st.markdown("Upload files (**PDF, Markdown, HTML, Plain Text**). The pipeline will parse document structures, apply your selected chunking strategy, generate 768-dim embeddings via local Ollama `nomic-embed-text`, prune near-duplicates, and dual-index into ChromaDB and BM25.")

    up_col1, up_col2 = st.columns([2, 1])

    with up_col1:
        uploaded_files = st.file_uploader(
            "Select files from your computer:",
            type=["pdf", "md", "txt", "html", "htm"],
            accept_multiple_files=True,
            help="Supported formats: PDF (.pdf), Markdown (.md), Plain Text (.txt), HTML (.html, .htm)"
        )

    with up_col2:
        chunk_strat = st.selectbox(
            "Chunking Strategy",
            ["recursive", "fixed_size", "semantic"],
            help="Recursive is recommended for Markdown and structured technical documentation with section headers."
        )

        st.markdown(f"""
        <div class="strategy-card">
            <b style="color: #38bdf8;">Strategy: {chunk_strat.capitalize()}</b><br>
            <span style="font-size: 12px; color: #94a3b8;">
                {"Splits recursively on markdown & document headers to preserve semantic boundaries." if chunk_strat == "recursive" else
                 "Uses sliding fixed-token windows with 50-token overlap." if chunk_strat == "fixed_size" else
                 "Splits sentences based on embedding cosine similarity dropoffs."}
            </span>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_files:
        st.info(f"📁 Selected **{len(uploaded_files)}** file(s): " + ", ".join([f"`{f.name}`" for f in uploaded_files]))

        if st.button("🚀 Process & Dual-Index Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Preparing document streams...")
            progress_bar.progress(15)

            try:
                files_payload = []
                for uf in uploaded_files:
                    files_payload.append(
                        ("files", (uf.name, uf.getvalue(), uf.type or "application/octet-stream"))
                    )

                status_text.text("Computing embeddings with Ollama & indexing into ChromaDB + BM25...")
                progress_bar.progress(55)

                resp = httpx.post(
                    f"{API_BASE}/v1/upload",
                    files=files_payload,
                    data={"chunking_strategy": chunk_strat},
                    timeout=300.0,
                )

                progress_bar.progress(100)
                if resp.status_code == 200:
                    res_data = resp.json()
                    status_text.text("Indexing completed successfully!")
                    st.toast(f"Indexed {res_data['documents']} documents into {res_data['chunks_indexed']} chunks!", icon="🎉")
                    time.sleep(1.2)
                    st.rerun()
                else:
                    status_text.text("Upload failed.")
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                status_text.text("Exception encountered.")
                st.error(f"Upload failed: {e}")

    st.markdown("---")
    st.subheader("📦 Or Ingest Benchmark Corpus (500K+ Q&A Benchmark)")
    st.caption("Load a batch from the 500K+ benchmark dataset (Confluence, GitHub, Slack, Gmail, Jira).")

    b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
    with b_col1:
        bench_count = st.selectbox("Sample Batch Size", [25, 50, 100, 250], index=0)
    with b_col2:
        bench_strat = st.selectbox("Benchmark Chunking Strategy", ["recursive", "fixed_size", "semantic"])
    with b_col3:
        st.write("")
        st.write("")
        bench_ingest = st.button("📥 Ingest Benchmark Batch", use_container_width=True)

    if bench_ingest:
        with st.spinner(f"Ingesting {bench_count} benchmark documents..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/v1/ingest",
                    json={"max_docs": int(bench_count), "chunking_strategy": bench_strat},
                    timeout=300.0,
                )
                if resp.status_code == 200:
                    d = resp.json()
                    st.toast(f"Ingested {d['documents']} docs into {d['chunks_indexed']} chunks!", icon="✅")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Failed: {e}")


# =========================================================================
# TAB 3: Hybrid vs. Dense A/B Arena
# =========================================================================
with tab_compare:
    st.subheader("⚖️ Side-by-Side: Hybrid Search (RRF) vs. Pure Dense Vectors")
    st.markdown("Compare how combining dense semantic embeddings with sparse BM25 keyword matching fixes semantic drift, catches exact function names and identifiers, and improves factual retrieval confidence.")

    comp_query = st.text_input(
        "Enter question for side-by-side benchmark:",
        value=default_q or "What are the deployment guidelines and rate limits?",
        key="comp_query_input"
    )

    if st.button("⚡ Run Comparative Benchmark", type="primary"):
        with st.spinner("Running both retrieval pipelines in parallel..."):
            col_left, col_right = st.columns(2)

            # Hybrid Run
            with col_left:
                st.markdown("#### ⚡ Hybrid Search (Dense + BM25)")
                try:
                    t0 = time.time()
                    resp_h = httpx.post(
                        f"{API_BASE}/v1/ask",
                        json={
                            "question": comp_query,
                            "retrieval_mode": "hybrid",
                            "use_reranker": False,
                            "dense_weight": dense_weight,
                            "sparse_weight": sparse_weight
                        },
                        timeout=120.0
                    )
                    t_h = time.time() - t0
                    if resp_h.status_code == 200:
                        data_h = resp_h.json()
                        st.markdown(f"> {data_h['answer']}")
                        st.metric("Hybrid Confidence", f"{data_h['confidence']['composite']:.1%}", f"Latency: {t_h:.2f}s")
                        st.caption(f"Retrieved Passages: {len(data_h['context_chunks'])}")
                    else:
                        st.error(resp_h.text)
                except Exception as e:
                    st.error(str(e))

            # Dense-Only Run
            with col_right:
                st.markdown("#### 🎯 Dense Vector Only (No BM25)")
                try:
                    t0 = time.time()
                    resp_d = httpx.post(
                        f"{API_BASE}/v1/ask",
                        json={
                            "question": comp_query,
                            "retrieval_mode": "dense",
                            "use_reranker": False
                        },
                        timeout=120.0
                    )
                    t_d = time.time() - t0
                    if resp_d.status_code == 200:
                        data_d = resp_d.json()
                        st.markdown(f"> {data_d['answer']}")
                        st.metric("Dense Confidence", f"{data_d['confidence']['composite']:.1%}", f"Latency: {t_d:.2f}s")
                        st.caption(f"Retrieved Passages: {len(data_d['context_chunks'])}")
                    else:
                        st.error(resp_d.text)
                except Exception as e:
                    st.error(str(e))


# =========================================================================
# TAB 4: Document Repository
# =========================================================================
with tab_docs:
    st.subheader("📚 Active Knowledge Base Documents")
    try:
        r = httpx.get(f"{API_BASE}/v1/documents", timeout=5.0)
        if r.status_code == 200:
            doc_data = r.json()
            total_docs = doc_data.get("total", 0)
            documents = doc_data.get("documents", [])

            col_stat1, col_stat2 = st.columns([1, 3])
            with col_stat1:
                st.metric("Total Indexed Documents", total_docs)

            if documents:
                table_rows = []
                for d in documents:
                    table_rows.append({
                        "Document ID": d["doc_id"],
                        "Title": d.get("title") or "Untitled",
                        "Source Format": d.get("source_type", "unknown").upper(),
                        "Characters": f"{d.get('char_count', 0):,}",
                    })
                st.dataframe(table_rows, use_container_width=True)
            else:
                st.info("No documents indexed yet. Use the 'Document Ingestion Studio' tab to upload documents.")
        else:
            st.error("Failed to load document catalog.")
    except Exception as e:
        st.error(f"Error connecting to backend API: {e}")
