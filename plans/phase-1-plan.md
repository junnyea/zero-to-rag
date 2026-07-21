# Phase 1 Implementation Plan: Classic RAG Baseline

This is the ordered implementation plan for Phase 1 of the Local Doc Q&A application, structured for our custom subagents and following the delegation protocols.

## 1. Objective
Build a fully local, single-user RAG baseline that enables users to upload `.txt`, `.md`, and `.pdf` files, chunk and embed them into a persistent Chroma collection using Ollama's `nomic-embed-text`, query them with `llama3.2:3b` with a strictly grounded prompt, and observe the effect of changing chunk sizes/overlaps on retrieval performance via a model-free `hit@3` evaluation script.

## 2. Tasks

### P1-T1: Initial Git Setup & Base Configuration (Size: S, Owner: Platform Engineer)
- **Description**: Initialize the git repository, set up a `.gitignore` to prevent committing virtual environments (`venv/`), caches (`__pycache__/`), and the persistent vector database (`chroma_db/`). Commit all baseline files.
- **Dependencies**: None.
- **Acceptance Criteria**:
  - [ ] Running `git status` shows a valid repository on the default branch.
  - [ ] `.gitignore` correctly ignores `chroma_db/`, `__pycache__/`, `.pytest_cache/`, and virtualenvs.
  - [ ] Initial files committed cleanly.
- **Files Touched**: `.gitignore`, new repository state.

### P1-T2: Project Scaffolding & Configuration Module (Size: S, Owner: Pipeline Engineer)
- **Description**: Set up the project directories (`rag/`, `eval/`, `sample_docs/`). Write the pinned dependencies in `requirements.txt`. Write the default configurations in `config.yaml`. Implement a configuration module `rag/config.py` using Python's `yaml` to load and validate configurations (especially asserting that overlap is less than half of chunk size).
- **Dependencies**: P1-T1.
- **Acceptance Criteria**:
  - [ ] `requirements.txt` contains pinned versions for `streamlit`, `langchain`, `langchain-ollama`, `langchain-chroma`, `chromadb`, `pypdf`, and `pytest`.
  - [ ] `config.yaml` exposes all parameters from Section 10 of the product requirements.
  - [ ] `rag/config.py` correctly parses and validates config, raising meaningful errors for invalid parameters (e.g. overlap >= chunk_size / 2).
- **Files Touched**: `requirements.txt`, `config.yaml`, `rag/config.py`.

### P1-T3: System Preflight Checks (Size: S, Owner: Pipeline Engineer)
- **Description**: Implement `rag/preflight.py` to verify the local Ollama daemon status. Ping `http://localhost:11434` and query `/api/tags` to check if `nomic-embed-text:latest` and `llama3.2:3b` are pulled. Expose a simple state interface for the Streamlit sidebar.
- **Dependencies**: P1-T2.
- **Acceptance Criteria**:
  - [ ] `preflight.py` correctly detects when Ollama is offline and returns a copy-paste instructions command (`ollama serve`).
  - [ ] Preflight correctly checks the availability of both required models.
  - [ ] Returns a clean status dictionary/object for UI consumption.
- **Files Touched**: `rag/preflight.py`.

### P1-T4: Ingestion Pipeline with Idempotency (Size: M, Owner: Pipeline Engineer)
- **Description**: Build the ingestion pipeline in `rag/ingest.py`. It must load `.txt`, `.md`, and `.pdf` files (using `pypdf` for PDFs). Split documents using LangChain's `RecursiveCharacterTextSplitter`. Prepend the required task prefix `search_document: ` to all chunk contents. Generate deterministic IDs for chunks using `sha256(file_bytes)[:12] + ":" + chunk_index`. Before writing new chunks for a file, delete any existing chunks matching `metadata.source == filename` to maintain strict idempotency. Store chunk details (source filename, chunk index, char span) in metadata. Save `embed_model`, `chunk_size`, and `chunk_overlap` in the collection's metadata.
- **Dependencies**: P1-T2.
- **Acceptance Criteria**:
  - [ ] `.txt`, `.md`, and `.pdf` files are successfully parsed and split.
  - [ ] Re-uploading a file deletes its old chunks and inserts new ones without duplication.
  - [ ] Chunk contents have the required `search_document: ` prefix prepended before being embedded.
  - [ ] Collection metadata stores the index parameters and triggers an error if a different embedder is configured.
- **Files Touched**: `rag/ingest.py`.

### P1-T5: Query Pipeline with Grounded Prompting (Size: M, Owner: Pipeline Engineer)
- **Description**: Build the querying pipeline in `rag/query.py`. Prepend `search_query: ` to the user's question. Embed and perform a similarity search on Chroma with cosine distance, retrieving `top_k` chunks. Implement the grounded QA prompt in `rag/prompts.py` forcing the model to answer ONLY from retrieved chunks, cite source filenames inline as `[filename]`, and refuse with *"I can't find that in the ingested documents."* if context is insufficient. Feed the prompt to `llama3.2:3b` via Ollama and support token streaming.
- **Dependencies**: P1-T2, P1-T4.
- **Acceptance Criteria**:
  - [ ] Queries are prefixed with `search_query: ` before embedding.
  - [ ] Prompt template successfully restrains the LLM to retrieved context.
  - [ ] Refusal string is returned exactly when context is empty or unrelated.
  - [ ] Source files are cited inline (e.g. `[handbook.pdf]`).
- **Files Touched**: `rag/query.py`, `rag/prompts.py`.

### P1-T6: Streamlit UI (Size: M, Owner: Pipeline Engineer)
- **Description**: Build `app.py` utilizing Streamlit. Features:
  - Tab 1 (Ingest): file uploader, "Ingest" button, progress bar, and index statistics card.
  - Tab 2 (Ask): text input for the question, "Ask" button, streamed token output, a sources line, and an expandable retrieved-context panel showing chunk text and similarity scores.
  - Sidebar: Sliders for `chunk_size`, `chunk_overlap`, and `top_k`. Model selectors. Preflight status component (green/red indicators with copy-paste instructions for red states). "Re-index all" and "Reset index" (with confirmation) buttons.
- **Dependencies**: P1-T3, P1-T4, P1-T5.
- **Acceptance Criteria**:
  - [ ] App boots without traceback and correctly displays the preflight statuses.
  - [ ] Changing parameters marks the index as stale in the UI, requiring "Re-index all".
  - [ ] Expandable retrieved-context panel displays retrieved chunks with their cosine similarity scores.
- **Files Touched**: `app.py`.

### P1-T7: Evaluation Harness & Chunk Experiment (Size: M, Owner: Eval Engineer)
- **Description**: Create sample documents in `sample_docs/` (e.g., HR handbook, company policies, FAQ). Write a gold QA test suite `eval/gold.jsonl` containing 8 QA pairs. Implement `eval/hit_at_k.py` which loads `eval/gold.jsonl`, retrieves chunks for each question, and calculates `hit@3` (percentage of questions where the correct file is retrieved). Conduct the chunking parameter sweep across `chunk_size ∈ {300, 800, 1500}` × `overlap ∈ {0, 150}`, log the results, and commit them.
- **Dependencies**: P1-T2, P1-T4, P1-T5.
- **Acceptance Criteria**:
  - [ ] `eval/hit_at_k.py` executes cleanly and outputs a neat metrics table.
  - [ ] Multi-parameter sweep is executed, showing how chunking affects retrieval.
  - [ ] Golden dataset and results are saved in the repo.
- **Files Touched**: `eval/gold.jsonl`, `eval/hit_at_k.py`, `sample_docs/`.

---

## 3. Waves & Execution Order

We will organize task implementation into 4 sequential waves of parallelized worktree execution:

- **Wave 1**:
  - Platform Engineer: **P1-T1** (Git Setup) [1pt]
- **Wave 2**:
  - Pipeline Engineer A: **P1-T2** (Scaffolding & Config) [1pt]
  - Pipeline Engineer B: **P1-T3** (Preflight Checks) [1pt]
- **Wave 3**:
  - Pipeline Engineer A: **P1-T4** (Ingestion Pipeline) [2pt]
  - Pipeline Engineer B: **P1-T5** (Query Pipeline) [2pt]
- **Wave 4**:
  - Pipeline Engineer: **P1-T6** (Streamlit UI) [2pt]
  - Eval Engineer: **P1-T7** (Evaluation Harness) [2pt]

---

## 4. Risks & Mitigations
1. **Ollama Connection**: Ollama could be offline or missing models on user machines.
   - *Mitigation*: Solved via Step 00 preflight check on application boot and sidebar warning.
2. **Duplicated Chunks**: Re-ingesting files could bloat the vector database and skew retrieval.
   - *Mitigation*: Maintain strict replace-on-reupload idempotency using `sha256` hashing and source metadata deletion before insert.
3. **Model Hallucinations**: Small local 3B model could fabricate answers for out-of-corpus queries.
   - *Mitigation*: Strictly configure the system prompt with zero-shot grounding instructions and a mandatory fallback string.
