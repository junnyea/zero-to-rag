# Implementation Plan — Stage 1: Classic RAG over Your Docs

This document evaluates the feasibility of the requirements in `product-req-desc.md` and provides a concrete step-by-step implementation plan for building the **Local Doc Q&A** application.

---

## 1. Feasibility Evaluation

We performed a technical feasibility assessment of the target environment and stack (Ollama + Streamlit + LangChain + Chroma DB). The results are highly favorable:

1. **Ollama Setup (Feasible & Complete):**
   - The Ollama server is running in the background.
   - The required models have been successfully downloaded and verified:
     - `nomic-embed-text` (Embedding model, 274 MB) — **Verified**
     - `llama3.2:3b` (LLM, 2.0 GB) — **Verified**
2. **Python Environment (Feasible & Set Up):**
   - The system Python environment is externally managed (PEP 668). We successfully bypass this by setting up a dedicated virtual environment (`venv/`).
   - All core Python dependencies have been successfully installed and verified in the virtual environment:
     - `streamlit` (v1.59.2)
     - `langchain` (v1.3.14)
     - `langchain-ollama` (v1.1.0)
     - `langchain-chroma` (v1.1.0)
     - `chromadb` (v1.5.9)
     - `pypdf` (v6.14.2)
     - `pyyaml` (v6.0.3)
3. **Hardware Constraints:**
   - The default model `llama3.2:3b` is lightweight and will run smoothly in this workspace.

---

## 2. Directory and File Layout

We will implement the following clean directory structure specified in the build plan:

```
/home/coder/workspace/zero-to-rag/
├── app.py                  # Streamlit frontend (tabs, sidebar, preflight checks)
├── config.yaml             # External application configuration defaults
├── requirements.txt        # Pinned project requirements (for reproducible setup)
├── venv/                   # Python virtual environment (already set up)
├── rag/
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration loading, validation, and schema definitions
│   ├── preflight.py        # Ollama server ping, pulled model validation
│   ├── ingest.py           # Ingestion pipeline (load -> split -> embed -> upsert with hash)
│   ├── query.py            # Query pipeline (embed query -> retrieve -> prompt LLM -> cite)
│   └── prompts.py          # Grounded QA prompt template
├── eval/
│   ├── gold.jsonl          # 8 gold Q&A pairs for retrieval quality testing
│   └── hit_at_k.py         # Evaluation script to compute hit@3 retrieval metrics
└── sample_docs/            # Directory containing test documents (TXT / MD / PDF)
```

---

## 3. Key Algorithmic Design Decisions

### A. Unique Hash-Based Chunk Replacement (Section 8.1)
*   **The Problem:** Re-uploading a file can cause duplicated chunks if we append blindly, bloating the index and polluting retrieval results.
*   **The Solution:**
    *   Compute the SHA-256 hash of the file bytes. Take the first 12 characters: `file_hash = sha256(file_bytes)[:12]`.
    *   For each chunk, construct a unique ID: `chunk_id = f"{file_hash}:{chunk_index}"`.
    *   To support replace-on-reupload, we will also track the original filename in the chunk's metadata (e.g., `source: "filename.txt"`).
    *   At ingestion time, before inserting any new chunks, we query Chroma to find any existing chunks with the same filename metadata and delete them using `.delete(where={"source": filename})` or delete by ID list. Then we insert the new chunks using their computed unique IDs.
    *   This ensures that editing or re-uploading a file never duplicates its content.

### B. Nomic Task Prefixes (ADR-3)
*   `nomic-embed-text` requires task-specific prefixes to produce high-quality embeddings:
    *   At ingestion time, each document chunk must be prepended with `"search_document: "`.
    *   At query time, the user's question must be prepended with `"search_query: "`.
*   We will ensure these prefixes are consistently applied on both sides of the pipeline, which significantly improves similarity score relevance.

### C. Grounded QA Prompt and Refusal Behavior (Section 8.2)
*   To prevent LLM hallucination and enforce high trust:
    *   We will supply the LLM with a strict prompt template in `rag/prompts.py`.
    *   The prompt will instruct the model to answer **only** using the retrieved chunks, cite the source files inline as `[filename]`, and if the context does not contain the answer, reply *exactly*: `"I can't find that in the ingested documents."`.
    *   We will evaluate the robustness of this refusal behavior using a dedicated question in our evaluation step.

### D. Stale Index Detection (Section 10 / Section 11)
*   Chroma persistent collection metadata will record the configuration parameters used at ingestion time: `embed_model`, `chunk_size`, and `chunk_overlap`.
*   At startup or query time, the app will read the collection's metadata. If the current `config.yaml` `embed_model` differs from the index's embedded model, we will raise a blocking warning in the UI requiring the user to run "Reset & re-index" to prevent dimension mismatch crashes.
*   If `chunk_size` or `chunk_overlap` differ, we will show a "stale index" warning indicating that the index parameters differ from current sliders, encouraging a re-index.

---

## 4. Detailed Step-by-Step Build Plan

### **Step 01: Create Sample Corpus**
*   Create a directory `sample_docs/`.
*   Generate 4+ rich test files (e.g., HR employee handbook, IT policy, corporate travel policy, and a company FAQ) written with realistic details to serve as our evaluation and demo base.
*   Files will be in plain text and Markdown formats.

### **Step 02: Scaffold App and Config**
*   Create `config.yaml` with the default settings (chunk size: 800, overlap: 150, top_k: 3, embed_model: nomic-embed-text, llm_model: llama3.2:3b).
*   Implement `rag/config.py` to parse and validate this YAML file.
*   Implement `rag/preflight.py` to:
    *   Ping `http://localhost:11434` to ensure Ollama is active.
    *   Query Ollama's local model list to ensure `nomic-embed-text` and `llama3.2:3b` are pulled.
    *   Expose this status cleanly for the UI.
*   Create `requirements.txt` with the installed versions.

### **Step 03: Build Ingestion Pipeline**
*   Implement `rag/ingest.py` containing the logic to:
    *   Load `.txt`, `.md`, and `.pdf` files.
    *   Split documents using LangChain's `RecursiveCharacterTextSplitter`.
    *   Compute the unique hash-based chunk IDs (`{file_hash}:{chunk_index}`).
    *   Remove any old chunks for the same filename from the Chroma collection.
    *   Prepend `"search_document: "` to the text of each chunk before embedding.
    *   Insert/upsert the chunks with their IDs, text, and metadata (source filename, chunk index, char span) into the persistent Chroma collection.
    *   Record ingestion parameters (`embed_model`, `chunk_size`, `chunk_overlap`) in Chroma collection metadata.

### **Step 04: Build Query Pipeline**
*   Implement `rag/query.py` containing the logic to:
    *   Embed the user query prepended with `"search_query: "` using the active embedding model.
    *   Query Chroma for the `top_k` (default 3) most similar chunks.
    *   Format the grounded QA prompt using retrieved chunks in `rag/prompts.py`.
    *   Query `llama3.2:3b` via LangChain's `ChatOllama` wrapper to generate the answer.
    *   Implement streamed generation (P1) and extract the text, source citations, and retrieved chunks with similarity scores.

### **Step 05: Implement Streamlit UI and Preflight**
*   Build `app.py` as the Streamlit entrypoint:
    *   **Sidebar:** Display preflight checkmarks (Ollama status, model availability). Show configuration sliders (`chunk_size`, `chunk_overlap`, `top_k`) and dropdowns for models. If config differs from index metadata, show a warning. Add a "Reset index" button.
    *   **Ingest Tab:** Display file uploader. Show index statistics card (total files, total chunks, active configuration used). Show ingest progress bar.
    *   **Ask Tab:** Provide a single-turn question box. On submit, stream the response. Display the final answer, followed by a list of cited files, and an expandable "Retrieved Context" panel displaying the raw chunks, filenames, and similarity scores sorted descending.
    *   Ensure all potential exceptions (Ollama down, empty index) display friendly banners with actionable copy-paste fixes instead of raw tracebacks.

### **Step 06: Build Evaluation Harness and Run Experiments**
*   Create `eval/gold.jsonl` containing 8 diverse Q&A pairs curated from the `sample_docs/` corpus, mapping questions to their expected source files.
*   Write `eval/hit_at_k.py` to:
    *   Programmatically query the Chroma database with the questions.
    *   Check if the expected source file is present in the top `k` chunks.
    *   Calculate and print the average hit@k score.
*   Perform a parameter grid search using `hit_at_k.py` over different combinations of `chunk_size` (300, 800, 1500) and `chunk_overlap` (0, 150).
*   Document the results and optimize the default parameters in `config.yaml` based on the empirical winner.

---

## 5. Verification Plan

We will perform automated and manual checks to verify correctness:
1.  **Replace Semantics Test:** Upload a sample file, inspect total chunks, make a small edit to the file, re-upload, and verify that the total chunk count is correctly updated without duplication.
2.  **Grounding & Refusal Test:** Ask a question completely unrelated to the sample documents (e.g., "What is the capital of France?") and verify that the app responds with exactly `"I can't find that in the ingested documents."` rather than generating world-knowledge.
3.  **Error Handling Drill:** Temporarily stop the Ollama server (or run on a wrong port) and confirm that the UI handles it gracefully with a red banner and instructions rather than crashing with a traceback.
4.  **Harness Run:** Execute `python eval/hit_at_k.py --k 3` and verify it reports the exact hit@3 retrieval accuracy metric.
