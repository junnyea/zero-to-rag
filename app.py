import os
import tempfile
import streamlit as st
from chromadb import PersistentClient
from rag.config import load_config, Settings
from rag.preflight import check_preflight
from rag.ingest import ingest_file, reset_index, EmbedModelMismatchError, get_chroma_collection_and_embeddings
from rag.query import query_rag

# Page setup
st.set_page_config(
    page_title="Local Doc Q&A",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Local Doc Q&A — Classic RAG over Your Docs")

# 1. Run Preflight Checks on boot/refresh
preflight_status = check_preflight()

# 2. Render Sidebar
st.sidebar.header("⚙️ Configuration")

# Render Preflight Status Block in Sidebar
st.sidebar.subheader("🔌 System Status")
all_green = True

# Ollama status
if preflight_status["ollama_online"]:
    st.sidebar.markdown("● **Ollama Daemon**: :green[Online]")
else:
    all_green = False
    st.sidebar.markdown("● **Ollama Daemon**: :red[Offline]")
    st.sidebar.code("ollama serve", language="bash")

# Embed model status
if preflight_status["embed_model_available"]:
    st.sidebar.markdown("● **Embedding Model**: :green[Ready]")
else:
    all_green = False
    st.sidebar.markdown("● **Embedding Model**: :red[Missing]")
    st.sidebar.code("ollama pull nomic-embed-text", language="bash")

# LLM model status
if preflight_status["llm_model_available"]:
    st.sidebar.markdown("● **LLM Model**: :green[Ready]")
else:
    all_green = False
    st.sidebar.markdown("● **LLM Model**: :red[Missing]")
    st.sidebar.code("ollama pull llama3.2:3b", language="bash")

# Sliders for parameters (loaded from config.yaml first, or managed in session state)
config = load_config()

# Chunking configs
st.sidebar.subheader("📝 Chunking Parameters")
chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=200,
    max_value=2000,
    value=config.chunk_size,
    step=100,
    help="Target character length of each document chunk"
)

# Overlap must be strictly less than chunk_size / 2
max_overlap = min(400, int(chunk_size / 2) - 10)
# Ensure default config overlap doesn't exceed newly set size boundaries
default_overlap = min(config.chunk_overlap, max_overlap)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=max_overlap,
    value=default_overlap,
    step=10,
    help="Number of overlapping characters between adjacent chunks"
)

# Retrieval config
st.sidebar.subheader("🔍 Retrieval Parameters")
top_k = st.sidebar.slider(
    "Top K Retrieved Chunks",
    min_value=1,
    max_value=10,
    value=config.top_k,
    step=1,
    help="Number of most similar chunks to retrieve as grounding context"
)

# Save current parameters back to config.yaml if they changed
if (chunk_size != config.chunk_size or
    chunk_overlap != config.chunk_overlap or
    top_k != config.top_k):
    # Overwrite config.yaml
    with open("config.yaml", "w", encoding="utf-8") as f:
        f.write(f"""# Default configurations for Local Doc Q&A RAG Pipeline (Phase 1)
chunk_size: {chunk_size}
chunk_overlap: {chunk_overlap}
top_k: {top_k}
embed_model: "search_document" # Placeholder or actual nomic
embed_model: "nomic-embed-text"
llm_model: "llama3.2:3b"
persist_dir: "./chroma_db"
""")

# Determine if index is stale by comparing sliders with database metadata
index_stale = False
is_empty_index = True
db_metadata = {}

if all_green:
    try:
        client = PersistentClient(path=config.persist_dir)
        if "docs" in [c.name for c in client.list_collections()]:
            collection = client.get_collection("docs")
            db_metadata = collection.metadata or {}

            # Check if there are any documents in the index
            count = collection.count()
            if count > 0:
                is_empty_index = False

            # If metadata exists, compare chunk params
            if db_metadata:
                stored_size = db_metadata.get("chunk_size")
                stored_overlap = db_metadata.get("chunk_overlap")
                stored_embed = db_metadata.get("embed_model")

                if (stored_size != chunk_size or
                    stored_overlap != chunk_overlap or
                    stored_embed != config.embed_model):
                    index_stale = True
    except Exception:
        pass

if index_stale:
    st.sidebar.warning("⚠️ **Index parameters out of sync!** Current sliders do not match the parameters of the active index. Please 'Reset Index' or re-ingest all files to apply new settings.")

# Disable UI buttons if system is offline
if not all_green:
    st.error("🚨 **System checks failed!** Please resolve the preflight instructions in the sidebar before running the application.")

# Tabs Setup
tab_ingest, tab_ask = st.tabs(["📂 Ingest Documents", "❓ Ask Questions"])

# TAB 1: INGEST
with tab_ingest:
    st.header("Upload and Index Your Corpus")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload files (.txt, .md, .pdf)",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            disabled=not all_green
        )

        if st.button("🚀 Ingest Uploaded Documents", disabled=not all_green or not uploaded_files):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            success_count = 0
            total_chunks = 0

            for i, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                status_text.markdown(f"Processing: **{filename}**...")

                # Write to temp file
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_path = temp_file.name

                try:
                    stats = ingest_file(temp_path)
                    # Correct filename in stats since temp_path has random name
                    stats["file"] = filename

                    success_count += 1
                    total_chunks += stats["chunks"]
                    st.success(f"Successfully indexed **{filename}** ({stats['chunks']} chunks created).")
                except EmbedModelMismatchError as e:
                    st.error(f"Error indexing {filename}: {e}")
                    break
                except Exception as e:
                    st.error(f"Error indexing {filename}: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                # Update progress
                progress_bar.progress(float(i + 1) / len(uploaded_files))

            status_text.markdown(f"🎉 **Ingestion complete!** Loaded {success_count}/{len(uploaded_files)} files. Total active chunks in index: {total_chunks}.")
            # Trigger refresh of DB stats
            st.rerun()

    with col2:
        st.subheader("📊 Index Statistics")
        if all_green:
            try:
                client = PersistentClient(path=config.persist_dir)
                if "docs" in [c.name for c in client.list_collections()]:
                    collection = client.get_collection("docs")
                    num_chunks = collection.count()

                    # Fetch all distinct filenames
                    all_metadatas = collection.get(include=["metadatas"])
                    metadatas_list = all_metadatas.get("metadatas", []) or []
                    filenames = list(set([m.get("source", "unknown") for m in metadatas_list if m]))

                    st.metric("Total Indexed Chunks", num_chunks)
                    st.metric("Unique Files Loaded", len(filenames))

                    if filenames:
                        st.markdown("**Active Files List:**")
                        for name in sorted(filenames):
                            st.markdown(f"- 📄 `{name}`")

                    # Display active index parameters
                    st.markdown("---")
                    st.markdown("**Active Index Parameters:**")
                    st.markdown(f"- **Embedding Model:** `{db_metadata.get('embed_model', config.embed_model)}`")
                    st.markdown(f"- **Chunk Size:** `{db_metadata.get('chunk_size', 'N/A')}`")
                    st.markdown(f"- **Chunk Overlap:** `{db_metadata.get('chunk_overlap', 'N/A')}`")
                else:
                    st.info("No active vector collection found. Please upload and ingest documents first.")
            except Exception as e:
                st.error(f"Error loading index stats: {e}")
        else:
            st.info("System is offline.")

        # Danger Zone / Reset Controls
        st.markdown("---")
        st.subheader("⚠️ Danger Zone")

        # Confirmation flag in session state
        if "confirm_reset" not in st.session_state:
            st.session_state.confirm_reset = False

        if not st.session_state.confirm_reset:
            if st.button("🗑️ Reset and Clear Vector Index", disabled=not all_green or is_empty_index):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.warning("🚨 **Are you absolutely sure?** This will permanently delete the active Chroma collection and all document chunks.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("🔥 Yes, Reset Index"):
                    try:
                        reset_index()
                        st.success("Successfully reset the index!")
                    except Exception as e:
                        st.error(f"Failed to reset index: {e}")
                    st.session_state.confirm_reset = False
                    st.rerun()
            with col_no:
                if st.button("❌ No, Cancel"):
                    st.session_state.confirm_reset = False
                    st.rerun()

# TAB 2: ASK
with tab_ask:
    st.header("Ask Questions Grounded in Your Documents")

    if is_empty_index:
        st.info("👉 **Please Ingest documents first!** Go to the 'Ingest Documents' tab, upload some text, md, or pdf files, and click 'Ingest' to start asking questions.")
    else:
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the company policy on remote work?",
            disabled=not all_green
        )

        if st.button("🔍 Ask", disabled=not all_green or not question):
            try:
                # Perform the query with scores
                stream_generator, retrieved_chunks = query_rag(question)

                st.subheader("💬 Answer")
                # Stream the response natively in Streamlit!
                full_response = st.write_stream(stream_generator)

                # Report sources if found in citations
                st.markdown("---")
                st.subheader("📚 Referenced Sources")

                # Extract distinct source filenames that appear in chunks
                sources = sorted(list(set([chunk["source"] for chunk in retrieved_chunks])))
                if sources:
                    cols_sources = st.columns(len(sources))
                    for idx, src in enumerate(sources):
                        with cols_sources[idx]:
                            st.info(f"📄 **{src}**")
                else:
                    st.markdown("*No references cited.*")

                # Retrieved Context expander
                st.subheader("🔍 Retrieval Transparency")
                with st.expander("Explore Retrieved Context Chunks"):
                    st.markdown("The following chunks were retrieved from Chroma using **cosine similarity** to ground the response:")

                    for idx, chunk in enumerate(retrieved_chunks):
                        score_color = ":green" if chunk["score"] >= 0.7 else (":orange" if chunk["score"] >= 0.5 else ":red")

                        st.markdown(f"### Chunk {idx + 1} — `{chunk['source']}` (Similarity: {score_color}[{chunk['score']}])")
                        st.markdown(f"**Index:** `{chunk['chunk_index']}` | **Char Span:** `{chunk['char_span']}`")
                        st.code(chunk["content"], language="text")
                        st.markdown("---")
            except Exception as e:
                st.error(f"Failed to execute query: {e}")
