import os
import json
from datetime import datetime
import streamlit as st
from typing import Dict, Any, List

from rag.config import load_config, save_config, validate_config
from rag.preflight import check_ollama_status, check_cohere_key_status
from rag.ingest import ingest_file, get_index_stats, reset_index, load_file_content
from rag.query import ask_question_stream
from rag.tracing import load_all_traces, TraceEmitter

# --- Page Setup ---
st.set_page_config(
    page_title="Local Doc Q&A",
    page_icon="🤖",
    layout="wide"
)

# Initialize Session States
if "show_reset_confirm" not in st.session_state:
    st.session_state.show_reset_confirm = False
if "current_trace_events" not in st.session_state:
    st.session_state.current_trace_events = None
if "selected_trace_id" not in st.session_state:
    st.session_state.selected_trace_id = None
if "cohere_calls" not in st.session_state:
    st.session_state.cohere_calls = 0

# Ensure sample_docs folder exists
SAMPLE_DOCS_DIR = "sample_docs"
if not os.path.exists(SAMPLE_DOCS_DIR):
    os.makedirs(SAMPLE_DOCS_DIR)

# --- Trace Timeline Renderer ---
def render_trace_timeline(events: List[Dict[str, Any]]):
    """Renders a simplified, highly readable timeline of trace events."""
    if not events:
        st.info("No events in this trace.")
        return

    # Sort events by sequence to be absolutely sure
    sorted_events = sorted(events, key=lambda x: x.get("seq", 0))

    for ev in sorted_events:
        phase = ev.get("phase", "REASON")
        step = ev.get("step", "query")
        detail = ev.get("detail", "")
        duration = ev.get("duration_ms")
        payload = ev.get("payload", {})

        # Determine emoji & styling based on phase
        if phase == "REASON":
            emoji = "🧠"
            color = "blue"
        elif phase == "ACT":
            emoji = "⚡"
            color = "orange"
        else: # OBSERVE
            emoji = "👁️"
            color = "green"

        # Duration label if present
        dur_str = f" :grey[({duration} ms)]" if duration is not None else ""

        # Title line
        st.markdown(f"{emoji} **:{color}[{phase}]** | `[{step}]` {detail}{dur_str}")

        # If payload has content, show it in an expander
        if payload and any(payload.values()):
            with st.expander("📝 View Details / Payload"):
                st.json(payload)

        st.markdown("---")

# --- Configuration & Preflight ---
config = load_config()
preflight = check_ollama_status()
cohere_preflight = check_cohere_key_status()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ RAG Configuration")

    # 1. Preflight Checklist
    st.subheader("🔍 Preflight Check")

    # Server Status
    if preflight["server_running"]:
        st.markdown("● **Ollama Server:** :green[Running]")
    else:
        st.markdown("● **Ollama Server:** :red[Offline]")
        st.error("Please run `ollama serve` in a separate terminal before launching the app.")
        st.code("ollama serve", language="bash")

    # Embed Model Status
    embed_status = preflight["models_status"]["embed_model"]
    embed_model_name = config.get("embed_model", "nomic-embed-text")
    if embed_status["status"]:
        st.markdown(f"● **Embedding Model:** :green[Ready] (`{embed_model_name}`)")
    else:
        st.markdown(f"● **Embedding Model:** :red[Missing] (`{embed_model_name}`)")
        st.warning(embed_status["message"])
        st.code(embed_status["command"], language="bash")

    # Chat Model Status
    llm_status = preflight["models_status"]["llm_model"]
    llm_model_name = config.get("llm_model", "llama3.2:3b")
    if llm_status["status"]:
        st.markdown(f"● **LLM Model:** :green[Ready] (`{llm_model_name}`)")
    else:
        st.markdown(f"● **LLM Model:** :red[Missing] (`{llm_model_name}`)")
        st.warning(llm_status["message"])
        st.code(llm_status["command"], language="bash")

    # Cohere Key Status
    if cohere_preflight["key_present"]:
        st.markdown("● **Cohere API Key:** :green[Loaded]")
    else:
        st.markdown("● **Cohere API Key:** :orange[Missing]")

    # Flag indicating whether we can run RAG
    preflight_ok = preflight["server_running"] and embed_status["status"] and llm_status["status"]

    st.divider()

    # 2. Strategy & Reranker Choice
    st.subheader("🎯 Strategy & Reranker")

    strategy_options = ["adaptive", "plain", "multi_query", "hyde"]
    strategy_idx = strategy_options.index(config.get("retrieval_strategy", "adaptive"))
    new_strategy = st.selectbox(
        "Retrieval Strategy",
        options=strategy_options,
        index=strategy_idx,
        help="adaptive: score-gated rewrite; plain: single vector search; multi_query: LLM expansion; hyde: hypothetical answer vector."
    )

    reranker_options = ["cohere", "none", "local"]
    reranker_idx = reranker_options.index(config.get("reranker", "cohere"))
    new_reranker = st.selectbox(
        "Reranker",
        options=reranker_options,
        index=reranker_idx,
        help="Select cross-encoder reranker to sort top chunks."
    )

    # UI Badges & Key Warnings for Cohere Egress (R3 & R6)
    if new_reranker == "cohere":
        st.warning("⚠️ **Data Egress Active:** Document content and queries leave this machine and are sent to Cohere's servers.")

        if not cohere_preflight["key_present"]:
            st.error("🔑 **API Key Required:** Cohere Reranker requires the `COHERE_API_KEY` environment variable.")
            st.code("export COHERE_API_KEY=\"your_key_here\"", language="bash")

        # Display quota usage counter (R3)
        st.metric("Cohere API Calls (Session)", f"{st.session_state.cohere_calls}", help="Calculated based on actual calls made this session.")

    st.divider()

    # 3. Sliders & Configuration Controls
    st.subheader("🛠️ Tuning Parameters")

    # Chunk Size
    new_chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=200,
        max_value=2000,
        value=int(config.get("chunk_size", 800)),
        step=100,
        help="Target length of each text segment."
    )

    # Chunk Overlap
    new_chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=400,
        value=int(config.get("chunk_overlap", 150)),
        step=10,
        help="Overlap size between adjacent segments to prevent splitting sentences."
    )

    # Top K (final context size)
    new_top_k = st.slider(
        "Retrieval Depth (Top-K)",
        min_value=1,
        max_value=10,
        value=int(config.get("top_k", 3)),
        step=1,
        help="Number of chunks to send to the final LLM prompt context."
    )

    # Candidate K (pool size for reranking)
    new_candidate_k = st.slider(
        "Candidate Pool (Candidate-K)",
        min_value=5,
        max_value=50,
        value=int(config.get("candidate_k", 20)),
        step=5,
        help="Size of the document pool retrieved before reranking."
    )

    # Rewrite Similarity Trigger
    new_rewrite_trigger_score = st.slider(
        "Rewrite Trigger Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(config.get("rewrite_trigger_score", 0.5)),
        step=0.05,
        help="Similarity score gate. If the top vector match falls below this, the LLM will rewrite the query (adaptive strategy only)."
    )

    # Max Rewrites
    new_max_rewrites = st.slider(
        "Max Query Rewrites",
        min_value=0,
        max_value=3,
        value=int(config.get("max_rewrites", 2)),
        step=1,
        help="Maximum re-try attempts for score-gated query rephrasing."
    )

    # Update active config dict
    temp_config = config.copy()
    temp_config["retrieval_strategy"] = new_strategy
    temp_config["reranker"] = new_reranker
    temp_config["chunk_size"] = new_chunk_size
    temp_config["chunk_overlap"] = new_chunk_overlap
    temp_config["top_k"] = new_top_k
    temp_config["candidate_k"] = new_candidate_k
    temp_config["rewrite_trigger_score"] = new_rewrite_trigger_score
    temp_config["max_rewrites"] = new_max_rewrites

    # Validate the modified values
    validation_err = validate_config(temp_config)
    if validation_err:
        st.error(validation_err)
        # Revert to original loaded settings to prevent writing broken values
        temp_config = config.copy()
    else:
        # Save config if changed
        if (config.get("retrieval_strategy") != new_strategy or
            config.get("reranker") != new_reranker or
            config.get("chunk_size") != new_chunk_size or
            config.get("chunk_overlap") != new_chunk_overlap or
            config.get("top_k") != new_top_k or
            config.get("candidate_k") != new_candidate_k or
            config.get("rewrite_trigger_score") != new_rewrite_trigger_score or
            config.get("max_rewrites") != new_max_rewrites):
            save_config(temp_config)
            config = temp_config

    st.divider()

    # 3. Index Stats & Parameter Mismatch Alerts
    stats = get_index_stats(config["persist_dir"])

    if stats["total_chunks"] > 0:
        # Evaluate compatibility of active sliders with built index
        stored_embed = stats["embed_model"]
        stored_size = stats["chunk_size"]
        stored_overlap = stats["chunk_overlap"]

        # A. Embed model mismatch (Hard Block)
        if stored_embed and stored_embed != config["embed_model"]:
            st.error(
                f"🚨 **Index Mismatch!** The existing index is embedded using `{stored_embed}`, "
                f"but current configuration requires `{config['embed_model']}`. "
                "Querying and ingestion are blocked. Please click 'Reset Index' to continue."
            )
            preflight_ok = False

        # B. Chunk parameters mismatch (Stale Warning)
        elif (stored_size != config["chunk_size"]) or (stored_overlap != config["chunk_overlap"]):
            st.warning(
                "⚠️ **Index is Stale!** The index was built with parameters: "
                f"Size={stored_size}, Overlap={stored_overlap}. "
                f"Current config is: Size={config['chunk_size']}, Overlap={config['chunk_overlap']}. "
                "Answers might be inconsistent. Please re-index."
            )

    st.divider()

    # 4. Operations (Reset Index)
    st.subheader("💥 Index Actions")
    if not st.session_state.show_reset_confirm:
        if st.button("Delete Index & DB", type="secondary", use_container_width=True):
            st.session_state.show_reset_confirm = True
            st.rerun()
    else:
        st.error("Are you sure you want to delete the index database? This cannot be undone.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Reset", type="primary", use_container_width=True):
                reset_index(config["persist_dir"])
                st.session_state.show_reset_confirm = False
                st.success("Database reset successfully.")
                st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_reset_confirm = False
                st.rerun()

# ----------------- MAIN VIEW -----------------
st.title("🤖 Local Doc Q&A")
st.markdown(
    "A privacy-first, fully local RAG system running on **LangChain, Chroma, and Ollama**."
)

if not preflight_ok:
    st.error("🔴 **Service Blocked:** Preflight checks are failing. Please check the sidebar to resolve active issues.")
    st.stop()

# Load latest index stats
stats = get_index_stats(config["persist_dir"])

# Create Ingest / Ask / Trace / Explorer tabs
tab1, tab2, tab3, tab4 = st.tabs(["📁 Ingest Documents", "❓ Ask Questions", "🧠 Decision Traces", "🗃️ Chroma DB Explorer"])

# ----------------- TAB 1: INGEST -----------------
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files to ingest (.txt, .md, .pdf)",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            help="Files uploaded here will be saved to sample_docs/ and ingested into the vector index."
        )

        ingest_clicked = st.button("Ingest Uploaded Files", type="primary")

        if ingest_clicked and uploaded_files:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            total_files = len(uploaded_files)
            ingest_results = []

            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                status_text.text(f"Processing: {filename}...")

                # Save file to sample_docs/
                save_path = os.path.join(SAMPLE_DOCS_DIR, filename)
                try:
                    file_bytes = uploaded_file.read()
                    with open(save_path, "wb") as f:
                        f.write(file_bytes)

                    # Load text content
                    file_content = load_file_content(save_path, filename)

                    # Ingest file into Chroma
                    res = ingest_file(file_content, filename, config)
                    ingest_results.append(res)
                except Exception as e:
                    st.error(f"Error processing `{filename}`: {e}")

                # Update progress
                progress_bar.progress((idx + 1) / total_files)

            status_text.text("Ingestion completed! Re-indexing statistics card...")
            st.success(f"Successfully processed {len(ingest_results)} files!")

            # Show written counts per file
            for res in ingest_results:
                st.markdown(f"✅ **{res['filename']}**: `{res['chunks_written']}` chunks written.")

            st.rerun()

        # Add "Re-index All" action
        st.subheader("🔄 Re-index Corpus")
        st.markdown(
            "Re-index all documents currently stored in `sample_docs/` using the active "
            f"sliders: **Size={config['chunk_size']}**, **Overlap={config['chunk_overlap']}**."
        )

        if st.button("Re-index All Local Docs", type="secondary"):
            local_files = [f for f in os.listdir(SAMPLE_DOCS_DIR) if os.path.isfile(os.path.join(SAMPLE_DOCS_DIR, f))]
            if not local_files:
                st.info("No files currently exist in `sample_docs/` to re-index.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                total_files = len(local_files)
                ingest_results = []

                for idx, filename in enumerate(local_files):
                    status_text.text(f"Re-indexing: {filename}...")
                    fpath = os.path.join(SAMPLE_DOCS_DIR, filename)
                    try:
                        file_content = load_file_content(fpath, filename)
                        res = ingest_file(file_content, filename, config)
                        ingest_results.append(res)
                    except Exception as e:
                        st.error(f"Error re-indexing `{filename}`: {e}")
                    progress_bar.progress((idx + 1) / total_files)

                status_text.text("Re-indexing completed!")
                st.success(f"Re-indexed {len(ingest_results)} files successfully.")
                st.rerun()

    with col_right:
        st.subheader("📊 Index Statistics")
        st.info("The statistics card reflects parameters used to build the currently persistent index.")

        if stats["total_chunks"] > 0:
            st.metric("Total Chunks", f"{stats['total_chunks']}")
            st.metric("Unique Files", f"{len(stats['unique_files'])}")

            st.markdown("**Active Index Parameters:**")
            st.markdown(f"- **Embedding Model:** `{stats['embed_model']}`")
            st.markdown(f"- **Chunk Size:** `{stats['chunk_size']}` chars")
            st.markdown(f"- **Chunk Overlap:** `{stats['chunk_overlap']}` chars")

            st.markdown("**File Manifest:**")
            for filename in stats["unique_files"]:
                st.markdown(f"- `{filename}`")
        else:
            st.warning("The index is currently empty. Please upload or place documents in `sample_docs/` to get started.")

# ----------------- TAB 2: ASK -----------------
with tab2:
    st.subheader("❓ Grounded Question & Answer")

    if stats["total_chunks"] == 0:
        st.info("👉 The index is empty. Please upload and ingest documents in the **Ingest Documents** tab before asking questions.")
    else:
        st.markdown(
            f"Ask single-turn questions grounded in the top `{config['top_k']}` retrieved chunks of your "
            f"**{len(stats['unique_files'])}** files. The LLM (`{config['llm_model']}`) will refuse to answer if the context does not contain the answer."
        )

        # User Question Input
        question = st.text_input(
            "Enter your question here:",
            placeholder="e.g. What is the standard core hours of operation?"
        )

        if st.button("Get Answer", type="primary") and question:
            with st.spinner("Retrieving document context and generating answer..."):
                # Call stream query (now returns trace object as 4th element)
                token_stream, chunks, status, active_trace = ask_question_stream(question, config)

                if status == "empty_index":
                    st.info("Please ingest documents first.")
                    st.session_state.current_trace_events = None
                elif status == "mismatch":
                    st.error("Model mismatch! Reset index and re-ingest first.")
                    st.session_state.current_trace_events = None
                elif status == "ok":
                    st.markdown("### 💬 Answer")

                    # Use streamlit stream visualization (P1 feature)
                    # We write streamed output in real-time as tokens generate
                    full_answer = st.write_stream(token_stream)

                    # Store the completed trace events into session state
                    st.session_state.current_trace_events = active_trace.events

                    # Highlight sources cited in the response
                    st.markdown("---")
                    st.markdown("#### 🔗 Sources Cited")
                    cited_files = []
                    for chunk in chunks:
                        source_file = chunk["source"]
                        # Match citation format like [filename.md] or [filename]
                        if f"[{source_file}]" in full_answer or source_file in full_answer:
                            cited_files.append(source_file)

                    if cited_files:
                        st.markdown(", ".join([f"`{f}`" for f in sorted(list(set(cited_files)))]))
                    else:
                        st.markdown("*No specific inline file citation matched or context insufficient.*")

                    # Expandable Context Panel
                    st.markdown("---")
                    with st.expander("🔍 Expanded Retrieved Context (Sorted by Similarity Score)"):
                        st.markdown(f"Retrieved the top `{len(chunks)}` most matching chunks:")

                        for idx, chunk in enumerate(chunks):
                            st.markdown(
                                f"##### Chunk {idx+1}: `{chunk['source']}` (Similarity Score: **{chunk['score']:.1%}**)"
                            )
                            st.code(chunk["content"], language="text")
                            st.markdown("---")

        # Inlined decision trace timeline under Tab 2 (even if we didn't just run a question, if session has it, render it)
        if st.session_state.current_trace_events:
            st.markdown("---")
            with st.expander("🧠 Active Decision Trace Timeline (Trace ID: " + st.session_state.current_trace_events[0].get("trace_id", "unknown") + ")", expanded=True):
                st.markdown("Below is the real-time reasoning and execution sequence for your query.")
                render_trace_timeline(st.session_state.current_trace_events)

# ----------------- TAB 3: TRACES -----------------
with tab3:
    st.subheader("🧠 Decision Trace Registry & Timeline")
    st.markdown(
        "Observe the exact step-by-step logic, reasoning, and actions taken by the RAG system. "
        "The system records a detailed trace log for every single query attempt."
    )

    all_traces = load_all_traces(config.get("trace_dir", "./traces"))

    if not all_traces:
        st.info("👉 No trace history found yet. Go to the **Ask Questions** tab and enter a question to generate a trace.")
    else:
        # Create option list for trace dropdown
        trace_options = []
        trace_id_map = {}

        for t in all_traces:
            tid = t["trace_id"]
            ts = t["timestamp"]
            # Convert ISO timestamp to a nice readable form
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = ts[:19].replace("T", " ")

            q = t.get("question") or "Empty or failed query"
            q_short = q if len(q) <= 60 else q[:57] + "..."
            label = f"[{time_str}] {q_short} ({tid})"

            trace_options.append(label)
            trace_id_map[label] = tid

        # Dropdown to select a trace
        selected_label = st.selectbox(
            "📋 Select past trace to inspect:",
            options=trace_options,
            help="Select any past question to inspect its step-by-step reasoning timeline."
        )

        selected_tid = trace_id_map[selected_label]

        # Load the selected trace data
        selected_trace = next((t for t in all_traces if t["trace_id"] == selected_tid), None)

        if selected_trace:
            # Display high-level metadata about this trace
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Trace ID:** `{selected_trace['trace_id']}`")
            with col2:
                st.markdown(f"**Retrieval Strategy:** `{selected_trace.get('strategy', 'plain')}`")
            with col3:
                st.markdown(f"**Reranker:** `{selected_trace.get('reranker', 'none')}`")

            # Trace download button (R8 / P1)
            trace_json_bytes = json.dumps(selected_trace["events"], indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Trace (JSON)",
                data=trace_json_bytes,
                file_name=f"trace_{selected_trace['trace_id']}.json",
                mime="application/json"
            )

            st.markdown("### 🗺️ Timeline of Decisions")
            render_trace_timeline(selected_trace["events"])

# ----------------- TAB 4: CHROMA EXPLORER -----------------
with tab4:
    st.subheader("🗃️ Chroma DB Vector Explorer")
    st.markdown(
        "Direct visual viewer into the persistent in-process **Chroma collection (`docs`)**. "
        "Browse the exact vector-indexed text chunks and metadata stored in your SQLite database."
    )

    if stats["total_chunks"] == 0:
        st.info("👉 The index is empty. Ingest documents first to view them here.")
    else:
        # Load raw data from Chroma
        from rag.ingest import get_chroma_client_and_collection
        try:
            client, collection = get_chroma_client_and_collection(config["persist_dir"])
            # Fetch all stored items
            raw_data = collection.get(include=["documents", "metadatas"])
            ids = raw_data.get("ids", [])
            documents = raw_data.get("documents", [])
            metadatas = raw_data.get("metadatas", [])

            # Compile into high-level structured list
            all_chunks = []
            for idx, cid in enumerate(ids):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                doc_text = documents[idx] if idx < len(documents) else ""

                # Extract original text if stored, else fallback
                original_text = meta.get("original_text") if meta else None
                if not original_text:
                    original_text = doc_text
                    if original_text.startswith("search_document: "):
                        original_text = original_text[len("search_document: "):]

                all_chunks.append({
                    "id": cid,
                    "source": meta.get("source", "unknown") if meta else "unknown",
                    "chunk_index": meta.get("chunk_index", -1) if meta else -1,
                    "char_start": meta.get("char_start", -1) if meta else -1,
                    "char_end": meta.get("char_end", -1) if meta else -1,
                    "file_hash": meta.get("file_hash", "unknown") if meta else "unknown",
                    "content": original_text
                })

            # Sort primarily by Source, then by Chunk Index
            all_chunks.sort(key=lambda x: (x["source"], x["chunk_index"]))

            # --- Filter Controls ---
            col1, col2 = st.columns([1, 2])
            with col1:
                # File Selector dropdown
                unique_sources = sorted(list(set(chunk["source"] for chunk in all_chunks)))
                file_filter = st.selectbox("📂 Filter by Source File:", ["All Files"] + unique_sources)

            with col2:
                # Keyword Search input
                search_query = st.text_input("🔍 Search Chunk Content (Keyword):", placeholder="Type a keyword to filter chunks...")

            # --- Apply Filters ---
            filtered_chunks = all_chunks
            if file_filter != "All Files":
                filtered_chunks = [c for c in filtered_chunks if c["source"] == file_filter]
            if search_query:
                filtered_chunks = [c for c in filtered_chunks if search_query.lower() in c["content"].lower()]

            # --- Display Results ---
            st.markdown(f"Showing **{len(filtered_chunks)}** of **{len(all_chunks)}** total chunks in the database:")

            for idx, chunk in enumerate(filtered_chunks):
                with st.container():
                    st.markdown(
                        f"🔹 **Chunk ID:** `{chunk['id']}` | **File:** `{chunk['source']}` | **Index:** `{chunk['chunk_index']}`"
                    )

                    # Display metadata fields in small text
                    st.caption(
                        f"File Hash: `{chunk['file_hash']}` | Characters: `{chunk['char_start']}` to `{chunk['char_end']}`"
                    )

                    # Text box displaying raw original text
                    st.code(chunk["content"], language="text")
                    st.divider()

        except Exception as e:
            st.error(f"Failed to read Chroma collection: {e}")

