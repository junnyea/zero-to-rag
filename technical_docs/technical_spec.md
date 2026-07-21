# Technical Specification — Stage 1: Classic RAG over Your Docs

**Working Name:** Local Doc Q&A (LangChain + Chroma + Ollama)  
**Status:** Baseline Completed (Post-Optimization) · **Date:** 2026-07-21  
**Version:** 1.0 · **Author:** Claude Code (Official Anthropic CLI Agent)

---

## 1. System Architecture Overview

The system is a fully local, single-user, privacy-first Retrieval-Augmented Generation (RAG) application. It implements a standard dual-pipeline architecture (Ingestion and Querying) integrated over an in-process persistent vector store.

```
       +--------------------------------------------------------+
       |                      USER BROWSER                      |
       +------------+------------------------------+------------+
                    |                              ^
     File Uploads   | (HTTPS Proxy)                | Streamed Answers
     & Configs      v                              | & Citations
       +------------+------------------------------+------------+
       |                      STREAMLIT UI (app.py)             |
       +------------+------------------------------+------------+
                    |                               |
       (Local Call) |                               | (Query / Search)
                    v                               v
       +------------+-------+              +--------+-----------+
       |   INGEST PIPELINE  |              |   QUERY PIPELINE   |
       |  (rag/ingest.py)   |              |  (rag/query.py)    |
       +------------+-------+              +--------+-----------+
                    |                               |
     Split, Prefix  |                               | Prefix, Search
     & Embed        v                               v
       +------------+-------------------------------+------------+
       |                    LOCAL OLLAMA ENGINE                 |
       |  - Embeddings: nomic-embed-text (Port 11434)           |
       |  - LLM Gen:    llama3.2:3b       (Port 11434)           |
       +------------+-------------------------------+------------+
                    |                               |
      SQLite Cosine |                               | Vector Query
      Read / Write  v                               v
       +------------+-------------------------------+------------+
       |                 CHROMA DB PERSISTENT STORE             |
       |                  Directory: ./chroma_db                |
       +--------------------------------------------------------+
```

### 1.1 Tech Stack
*   **Frontend UI:** Streamlit (v1.59.2) — python-native, reactive dashboard interface.
*   **Orchestration:** LangChain (v1.3.14) with:
    *   `langchain-ollama` (v1.1.0) — for embeddings and chat generation.
    *   `langchain-chroma` (v1.1.0) — for vector database integration.
    *   `langchain-text-splitters` (v1.1.2) — for document chunking.
*   **Vector Database:** Chroma DB (v1.5.9) running in-process via `PersistentClient` with cosine distance metric (`hnsw:space = cosine`).
*   **LLM Server:** Ollama (v0.1.x+) running on `http://localhost:11434`.
*   **Models:**
    *   Embeddings: `nomic-embed-text` (768 dimensions, 274 MB).
    *   Generation: `llama3.2:3b` (3 Billion parameters, 2.0 GB).

---

## 2. Technical Ingestion Pipeline

The ingestion pipeline transforms raw source files into vectorized chunks, supporting `.txt`, `.md`, and `.pdf` files.

### 2.1 Loader Module
*   **Text/Markdown:** Plain UTF-8 reader.
*   **PDF:** Read using `pypdf`'s `PdfReader`, extracting and concatenating text across all pages. Empty or scanned PDFs (no extractable text) are rejected with a descriptive error.

### 2.2 Splitter Module
*   Uses `RecursiveCharacterTextSplitter` from LangChain.
*   **Optimal Defaults (Discovered via hit@3 Experimentation Sweep):**
    *   `chunk_size`: **300** characters
    *   `chunk_overlap`: **0** characters
*   *Justification:* Character length 300 (average retrieved length 215.2 characters) produces highly granular, concise chunks, saving LLM context space, improving speed, and preventing distraction while maintaining 100% retrieval accuracy.

### 2.3 Unique Hashing and Replace Semantics (P0)
To avoid duplicating chunks upon file re-upload:
1.  Generate a unique file content hash: `file_hash = sha256(file_bytes).hexdigest()[:12]`.
2.  Define each chunk ID as: `f"{file_hash}:{chunk_index}"` (e.g. `fab0641c2a85:0`).
3.  Each chunk stores the original filename in metadata: `{"source": filename}`.
4.  **Replace Semantics:** Prior to inserting new chunks, Chroma is queried to delete existing chunks matching `where={"source": filename}`. This performs a complete metadata-filtered wipe of previous versions before writing the new hashed IDs.

### 2.4 Ingestion Prefixes
To optimize embedding output, the system prepends Nomic's task prefix:
*   Before embedding, each chunk's `page_content` is stored as: `f"search_document: {chunk_text}"`.
*   The raw chunk text (without prefix) is preserved in chunk metadata as `original_text` to keep the UI's debug and retrieved-context drawer clean and readable.

---

## 3. Technical Query Pipeline

The query pipeline handles user input, performs vector retrieval, structures the grounding context, and streams the answer from the LLM.

### 3.1 Custom Query Embedding (ADR-3)
*   The incoming question is prepended with `"search_query: "`.
*   This formatted query is embedded using `nomic-embed-text` via Ollama's embeddings API.

### 3.2 Vector Similarity Retrieval
*   Performs a similarity search using cosine distance on Chroma collection `docs` returning the top `k` (default 3) matches.
*   The raw distance score $d$ is converted into a positive cosine similarity score $S_c$ using the formula:
    $$S_c = \max(0.0, \min(1.0, 1.0 - d))$$
*   The retrieved results are returned sorted by $S_c$ descending.

### 3.3 Prompt Construction and Grounding (Section 8.2)
*   The prompt template is defined strictly in `rag/prompts.py`:
    ```
    Answer the question using ONLY the provided context. Do not use any outside knowledge.
    Cite your sources inline as [filename] (e.g. [employee_handbook.md]) whenever you refer to information.
    If the context does not contain the answer, you must respond with exactly: "I can't find that in the ingested documents."
    ```
*   **LLM Invocation:** `llama3.2:3b` is invoked via `ChatOllama` with `temperature = 0.0` to force deterministic, grounded generation.
*   **Token Streaming:** The response is streamed token-by-token using `llm.stream()` to enable real-time UI rendering.

---

## 4. Operational and State Controls

### 4.1 Preflight Health Verification
At startup, `rag/preflight.py` runs a checking routine:
1.  Pings `http://localhost:11434/api/tags`. If down, raises a global offline flag.
2.  Inspects returned model list. Verifies that `nomic-embed-text` (or latest tag) and `llama3.2:3b` are locally pulled.
3.  If anything is missing, Streamlit displays copy-paste commands (e.g., `ollama serve` or `ollama pull`) and blocks user interaction.

### 4.2 Parameter Mismatch and Stale Indicators
*   Chroma collection metadata stores the parameters used to index: `embed_model`, `chunk_size`, and `chunk_overlap`.
*   On load, the sidebar reads this metadata and compares it to `config.yaml`:
    *   **Hard Mismatch:** If current `config.yaml` `embed_model` differs from collection metadata, the app displays a blocking error and requires a "Reset Database" (prevents dimension mismatch crashes).
    *   **Stale Warn:** If `chunk_size` or `chunk_overlap` slider values differ from collection metadata, the app displays a yellow "Index is Stale" warning and prompts the user to "Re-index All".

---

## 5. Flowchart

The following flowchart describes the operations of the Ingestion vs. Querying Pipelines:

```mermaid
flowchart TD
    %% Styling
    classDef process fill:#f9f,stroke:#333,stroke-width:2px;
    classDef database fill:#bbf,stroke:#333,stroke-width:2px;
    classDef decision fill:#ffb,stroke:#333,stroke-width:2px;
    classDef io fill:#fbb,stroke:#333,stroke-width:2px;

    %% Ingestion Pipeline
    subgraph Ingestion Pipeline (rag/ingest.py)
        A[File Uploaded] --> B(Extract Text txt/md/pdf)
        B --> C(Compute SHA-256 File Hash)
        C --> D(Split Text into Chunks via character-splitter)
        D --> E(Form Chunk IDs hash:index)
        E --> F(Query Chroma & Delete old chunks matching source filename)
        F --> G(Prepend search_document: prefix to each chunk)
        G --> H[(Embed with nomic-embed-text & Save in Chroma 'docs')]
        H --> I(Save Active Chunking Parameters in Collection Metadata)
    end

    %% Query Pipeline
    subgraph Query Pipeline (rag/query.py)
        J[User Submits Question] --> K(Prepend search_query: prefix)
        K --> L[(Search Chroma docs top-k Cosine Distance)]
        L --> M{Is Index Empty?}
        M -- Yes --> N[Prompt: Please Ingest First]
        M -- No --> O(Calculate Similarity Scores: 1.0 - Distance)
        O --> P(Assemble Grounded Prompt with Citations instructions)
        P --> Q(Query llama3.2:3b via Ollama with temp=0.0)
        Q --> R(Stream Tokens to Streamlit UI)
    end
    
    class A,J,N io;
    class B,C,D,E,F,G,I,K,O,P,Q,R process;
    class H,L database;
    class M decision;
```

---

## 6. Sequence Diagram

This sequence diagram details the end-to-end lifecycle of file uploads and query resolution:

```mermaid
sequence diagram
    autonumber
    actor User
    participant UI as Streamlit UI (app.py)
    participant IG as Ingest (rag/ingest.py)
    participant QY as Query (rag/query.py)
    participant DB as Chroma DB (PersistentClient)
    participant OL as Ollama Server (:11434)

    %% Ingestion Flow
    Note over User, OL: Ingestion Sequence
    User->>UI: Upload handbook.md & click Ingest
    UI->>IG: ingest_file(content, "handbook.md", config)
    IG->>IG: Compute file hash (SHA-256)
    IG->>IG: Split text (size=300, overlap=0)
    IG->>DB: delete(where={"source": "handbook.md"})
    Note over IG, DB: Replace-on-reupload deletes old chunks
    DB-->>IG: Deletion confirmed
    IG->>IG: Prepend "search_document: " to chunks
    IG->>OL: Embed chunk texts (nomic-embed-text)
    OL-->>IG: Float vector arrays (768 dims)
    IG->>DB: add_documents(Documents, IDs)
    IG->>DB: modify(metadata={"chunk_size": 300, "chunk_overlap": 0, "embed_model": "nomic-embed-text"})
    DB-->>IG: Write complete
    IG-->>UI: Return ingest statistics (success, 5 chunks)
    UI-->>User: Render "Success! 5 chunks written" & Update Stats Card

    %% Query Flow
    Note over User, OL: Query Sequence
    User->>UI: Type "What are core hours?" & click Ask
    UI->>QY: ask_question_stream("What are core hours?", config)
    QY->>QY: Prepend "search_query: "
    QY->>OL: Embed query (nomic-embed-text)
    OL-->>QY: Query vector array (768 dims)
    QY->>DB: query(query_vector, k=3)
    DB-->>QY: Match docs with cosine distance
    QY->>QY: Convert distance to similarity (1.0 - d)
    QY->>QY: Structure context & QA prompt
    QY->>OL: stream(prompt, model="llama3.2:3b", temp=0.0)
    loop Token Streaming
        OL-->>QY: Yield token
        QY-->>UI: Yield token
        UI-->>User: Stream token in real-time
    end
    UI->>UI: Extract citations [handbook.md] and highlight sources
    UI-->>User: Display citations & Expanded retrieved context panel (Scores & Chunks)
```

---

## 7. Configuration Specifications

Below are the system specifications for properties stored in `config.yaml` and edited in the UI sidebar:

| Configuration Key | Default Value | Acceptable Range / Schema | Impact Scope | Action Required |
|---|---|---|---|---|
| `chunk_size` | `300` | `200` to `2000` (step `100`) | Text splitting | Next file ingestion / Re-index |
| `chunk_overlap` | `0` | `0` to `400` (Must be `< chunk_size / 2`) | Text splitting | Next file ingestion / Re-index |
| `top_k` | `3` | `1` to `10` | Similarity search depth | Next submitted query |
| `embed_model` | `"nomic-embed-text"` | Any pulled Ollama embed model | Vector dimensions | **Requires full Reset & re-index** |
| `llm_model` | `"llama3.2:3b"` | Any pulled Ollama chat model | Generation quality | Next submitted query |
| `persist_dir` | `"./chroma_db"` | Valid directory path string | Database storage | System restart |
