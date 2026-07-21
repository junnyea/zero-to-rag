# RAG Pipeline 3-Phase Build Prompts

This document contains the official prompts, specifications, and Definitions of Done (DoD) for the three phases of the production-grade local RAG pipeline.

---

## Phase 1: Classic RAG Baseline (Stage 1)

### Scope & Requirements
Implement a fully local, single-user RAG application using Streamlit, LangChain, Chroma, and Ollama.
1. **Models**: Use `nomic-embed-text` for embedding (prepend `search_document: ` for ingestion and `search_query: ` for query) and `llama3.2:3b` for completion.
2. **Ingestion**: Accept `.txt`, `.md`, and `.pdf` files. Split using `RecursiveCharacterTextSplitter`. Chunk IDs must be `sha256(file_bytes)[:12] + ":" + chunk_index` to support replace-on-reupload idempotency.
3. **Retrieval**: Retrieve top-k (default 3) chunks using Chroma persistent client with cosine distance. Store indexing parameters in collection metadata.
4. **Generation**: Implement a strict grounding prompt template forcing the model to answer only from the retrieved chunks, cite source filenames as `[filename]`, and refuse with *"I can't find that in the ingested documents."* if the context is insufficient.
5. **UI**: Built in Streamlit with two tabs (Ingest and Ask) and a configuration and preflight check sidebar. Show color indicators for Ollama server, embed model, and LLM model status.
6. **Evaluation**: Write `eval/hit_at_k.py` to calculate model-free `hit@3` on `eval/gold.jsonl` (contains 8 gold Q&A cases).

### Definition of Done (DoD)
- [ ] App boots and preflight sidebar correctly reports Ollama and model statuses.
- [ ] Multi-file ingestion is idempotent; re-uploading replaces chunks without duplicating.
- [ ] Grounded responses are strictly maintained; unknown questions result in the refusal string.
- [ ] Chunks and similarity scores are shown in an expandable UI section.
- [ ] `eval/hit_at_k.py` runs successfully, reporting baseline hit@3 performance.
- [ ] Core logic is covered by unit tests, all passing.
- [ ] Code is reviewed by `code-reviewer` with zero blocking findings.

---

## Phase 2: Rerank & Evaluation (Stage 2)

### Scope & Requirements
Upgrade the retrieval pipeline with a two-stage search using Cohere Rerank.
1. **Two-Stage Retrieval**: Over-retrieve top-50 candidates via dense search, then rerank them down to final top-6 using Cohere Rerank (e.g. `rerank-v3.5`), controlled by a `RERANK_ENABLED` flag.
2. **Relevance Threshold**: Discard chunks below a threshold $\tau$. If no chunks pass, route to the "insufficient context" refusal path.
3. **Resilience**: 500ms timeout and 2 retries with exponential backoff for Cohere API. Fall back to Phase 1 dense retrieval on failure, log the error, and tag the response as degraded.
4. **Golden Dataset**: Set up `eval/golden_dataset.jsonl` containing 150 questions across 5 tiers (factual lookup, multi-hop, comparison, procedural, unanswerable).
5. **Evaluation Harness**: Calculate Recall@5, Recall@20, MRR, NDCG@10, faithfulness, answer relevance, citation accuracy, p50/p95 latency, and cost per query.

### Definition of Done (DoD)
- [ ] Two-stage retrieval implemented and gate-controlled by `RERANK_ENABLED` flag.
- [ ] Cohere API timeout, retries, and graceful fallback to dense retrieval verified in unit tests.
- [ ] Golden dataset established and locked.
- [ ] Evaluation harness runs successfully, producing absolute metrics and Phase 1 vs Phase 2 delta tables.
- [ ] All code modifications cleared by `code-reviewer` with zero blocking findings.

---

## Phase 3: SRE & Production Platform (Stage 3)

### Scope & Requirements
Productionize the RAG pipeline as a high-performance, robust microservice.
1. **Service Layer**: FastAPI backend supporting streaming responses, request validation, API key/OIDC authentication, per-client rate limiting, and health/readiness endpoints.
2. **Observability**: Per-request trace capture (query, candidate IDs, scores, model versions, token counts, cost, latencies) exported to OTel/LangSmith/Langfuse. Dashboards for error rates, latencies, costs.
3. **Resilience**: Timeouts, circuit breakers, and degraded modes for all third-party services.
4. **Security**: Secrets managed in environment, prompt injection protection on retrieved content.
5. **Index Lifecycle**: Versioned indices, blue/green reindexing, incremental ingestion.
6. **CI/CD**: PRs run unit tests + frozen eval suite. 5% canary deployment.

### Definition of Done (DoD)
- [ ] FastAPI backend running with streaming, validation, authentication, and rate limiting.
- [ ] Observability traces and dashboard fully functional.
- [ ] Resiliency circuit breakers and fallback paths verified via chaos drills.
- [ ] Prompt injection protection block rate is $\ge 100\%$ on a 30-item red-team set.
- [ ] Runbook with 8 failure scenarios and end-to-end tested rollback procedure.
- [ ] High-load tests run to $2\times$ peak, meeting p95 latency SLOs.
- [ ] All code cleared by `code-reviewer` with zero blocking findings.
