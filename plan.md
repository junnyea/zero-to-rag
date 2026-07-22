# Implementation Plan — Stage 2: Adaptive Retrieval, Reranking & Decision Tracing

This document outlines the step-by-step technical plan to upgrade our local RAG application to **Stage 2: Adaptive Retrieval, Reranking, and Decision Tracing** in accordance with the PRD (`product-req-desc-stage2`).

---

## 1. Technical Architecture Overview

Stage 2 transforms our flat, single-path retrieval pipeline into an **adaptive, multi-strategy, reranked pipeline with 100% decision observability**.

```
                [ User Question ]
                        │
                        ▼
            [ Step 1: Retrieval Strategy ]
     ┌──────────────────┼──────────────────┬─────────────────┐
     ▼                  ▼                  ▼                 ▼
[ Adaptive ]         [ Plain ]       [ Multi-Query ]      [ HyDE ]
  Embed Q          Embed Q directly    LLM generates      LLM writes
     │                  │              3 query variants   hypothetical
     ▼                  ▼                  │              passage; embed it
Search Chroma      Search Chroma           ▼                 │
(candidate_k)       (candidate_k)     Search Chroma          ▼
     │                  │             for each variant  Search Chroma
     ▼                  │                  │            (candidate_k)
[ Similarity Gate ]     │                  ▼                 │
Top Score < 0.5?        │            Union-deduplicate       │
  ├── Yes: Rewrite      │            candidate pools         │
  │   (Max 2 retries)   │                  │                 │
  └── No: Proceed       │                  │                 │
     │                  │                  │                 │
     └──────────────────┼──────────────────┴─────────────────┘
                        │
                        ▼
                [ Candidate Pool ] (Up to candidate_k chunks)
                        │
                        ▼
             [ Step 3: Reranker Choice ]
                        ├── None   ──► Vector-order Top-K
                        └── Cohere ──► Cohere Rerank API (20 -> Top-K)
                             │           (Fallback to Vector-order if error)
                             ▼
                [ Final Context Chunks ] (Top-K)
                        │
                        ▼
              [ Step 4: LLM Generation ]
                    (Streamed Answer)
```

### Key Design Decisions & Code Changes
1.  **Observability First:** We will build a central `TraceEmitter` class. Every step of the query path must emit structured trace events representing the **REASON → ACT → OBSERVE** pattern. These events are saved locally in JSONL format to `traces/YYYY-MM-DD.jsonl` and rendered in the UI.
2.  **No Egress Without Consent:** Since Cohere Rerank sends document snippets to a third-party cloud API, we require an explicit key check and showcase a persistent **"Sends query + candidate chunks to Cohere"** badge in the UI whenever Cohere is selected.
3.  **Strict Fallbacks (Robustness):** Missing Cohere API keys, network dropouts, or rate limits must be gracefully caught. The system will fall back to vector search top-3, log the fallback reason in the trace, display a warning in the UI, and continue answering without throwing stack traces.
4.  **No Ingestion Code Changes:** The vector database collection format, chunk hashes, and embedding models remain fully compatible with Stage 1. All upgrades are in the *query path*.

---

## 2. Updated Directory Layout

```
/home/coder/workspace/zero-to-rag/
├── app.py                      # Updated: Sidebar strategy selectors, Trace visualizer, Egress badges
├── config.yaml                 # Updated: defaults for Stage 2 config parameters
├── requirements.txt            # Updated: cohere, sentence-transformers (for local reranker)
├── traces/                     # New: Local JSONL persistent trace directory (Y-M-D.jsonl)
├── rag/
│   ├── __init__.py
│   ├── config.py               # Updated: Load/save/validate new configuration parameters
│   ├── preflight.py            # Updated: Added Cohere API Key / status check
│   ├── ingest.py
│   ├── query.py                # Updated: Multi-strategy retrievers, rewrite loops, rerankers
│   ├── prompts.py              # Updated: Prompt templates for Rewriting, Multi-Query, and HyDE
│   └── tracing.py              # New: Structured TraceEmitter core, local storage, and schema
├── eval/
│   ├── gold.jsonl              # Updated: 4 new hard/oblique questions (R10)
│   └── hit_at_k.py
└── sample_docs/
```

---

## 3. Detailed Step-by-Step Build Plan

### **Phase 10 — Trace Core & Instrument Stage 1 Baseline**
We build the observability harness first so that all future feature steps are born fully instrumented.

1.  **Create `rag/tracing.py`:**
    *   Define the trace event schema version: `schema_version = 1`.
    *   Create a `TraceEmitter` class or helper methods to manage an active trace.
    *   Fields in a trace event: `{trace_id, seq, timestamp, phase: "REASON"|"ACT"|"OBSERVE", step, detail, payload, duration_ms}`.
    *   Implement saving traces to disk in `traces/YYYY-MM-DD.jsonl`. Maintain a memory or local file log of recent trace IDs to allow past-trace retrieval.
    *   Implement a rolling retention mechanism to keep the last `trace_keep` (default 200) traces (P1).
2.  **Instrument the Baseline Path in `rag/query.py`:**
    *   Ensure every query gets a unique `trace_id` (e.g. `tr_` + unique suffix).
    *   Add trace logging around baseline vector retrieval (REASON: why searching, ACT: similarity_search_with_score, OBSERVE: number of chunks found, top scores).
    *   Add trace logging around LLM generation (REASON: prompting LLM, ACT: chat model invocation, OBSERVE: response generated, duration).
3.  **Build the Streamlit Trace UI (Third Tab):**
    *   Create a dedicated "Trace Viewer" tab in `app.py`.
    *   Show a sidebar/dropdown of recent trace IDs. Selecting a trace reads its JSONL lines from disk.
    *   Render the timeline of the selected trace visually with colored icons representing **REASON** (e.g. 🧠), **ACT** (e.g. ⚡), and **OBSERVE** (e.g. 👁️), along with duration and expandable payloads (no graphs/DAGs, per non-goals).
    *   Also render the timeline of the active question inlined below the chat answer in the "Ask Questions" tab.

### **Phase 11 — Strategy & Reranker Selectors**
Add configuration plumbing and verify we can reproduce Stage 1 exactly as a regression baseline.

1.  **Update `config.yaml` and `rag/config.py`:**
    *   Add new keys: `retrieval_strategy`, `reranker`, `rerank_model`, `candidate_k`, `rewrite_trigger_score`, `max_rewrites`, `multi_query_n`, `query_llm_model`, `trace_dir`, `trace_keep`.
    *   Update `load_config`, `save_config`, and `validate_config` to handle these.
2.  **Add Sidebar Selectors in `app.py`:**
    *   Add Sidebar controls for Retrieval Strategy (adaptive, plain, multi_query, hyde) and Reranker (cohere, none, local).
    *   Add configuration tuning controls (P1): sliders for rewrite trigger score, max rewrites, and candidate pool size (`candidate_k`).
    *   Ensure sidebar modifications save to config or local state and take effect on the next question.
3.  **Implement the Baseline Regression Check:**
    *   Ensure when `strategy="plain"` and `reranker="none"`, the output matches Stage 1 vector-order top-3 retrieval exactly.
    *   Create a test scenario or run `eval/hit_at_k.py` to verify consistency.

### **Phase 12 — Cohere Rerank Integration**
Integrate the Cohere Rerank cloud endpoint with resilient fallbacks.

1.  **Install dependencies and setup preflight:**
    *   Add `cohere` to `requirements.txt`.
    *   Update `rag/preflight.py` to check for `COHERE_API_KEY` (read from `.env` or env vars). Report key presence status in sidebar checklist with setup/remediation guidelines.
2.  **Implement Cohere Rerank in `rag/query.py`:**
    *   When `reranker="cohere"`, fetch up to `candidate_k` (default 20) vector chunks.
    *   Format as list of texts. Call Cohere Rerank client using `cohere.ClientV2` or `cohere.Client` with model `rerank-v3.5`.
    *   Trace the call: REASON: why calling (sending N chunks to model), ACT: API call, OBSERVE: reranker outputs with scores, and the final top_k selected.
    *   Implement fallback block: wrap in try/except. If the API key is missing or invalid, or there is a network error, log a `REASON` for fallback in the trace, show a warning toast/banner in the UI, and return the top_k chunks by raw vector score. No stack traces.
3.  **Add Quota Guard & Egress Badge:**
    *   In the sidebar, if `reranker="cohere"` is selected, display an unmissable orange badge: **⚠️ Data Egress Active (Sends chunks to Cohere)**.
    *   Keep a thread/session-safe call counter for Cohere calls. Show the monthly quota status / session counter in the sidebar.

### **Phase 13 — Adaptive Query-Rewrite Loop**
Implement the core adaptive score-gated query rewriting.

1.  **Create prompts in `rag/prompts.py`:**
    *   Add `QUERY_REWRITE_PROMPT`: Instructs the LLM to rewrite an input question into a single search-optimized query.
2.  **Implement Score-Gated Loop in `rag/query.py`:**
    *   Retrieve initial chunks using `candidate_k`. Check the top chunk's vector similarity score.
    *   If score >= `rewrite_trigger_score` (default 0.5), proceed to step 3 (Rerank).
    *   If score < `rewrite_trigger_score`, enter loop:
        *   Log REASON: "Top score < trigger → rewrite required (attempt 1/2)".
        *   ACT: Call local LLM (or `query_llm_model`) with `QUERY_REWRITE_PROMPT` to rephrase the question.
        *   Log OBSERVE: "Rewrote query to: [new_query]".
        *   ACT: Search Chroma using the rewritten query.
        *   Log OBSERVE: "Obtained N chunks, top similarity: X".
        *   Store all returned chunks from this search attempt in a candidate pool list.
        *   Check if top similarity >= threshold. If yes, exit early. If no, repeat up to `max_rewrites` (default 2).
    *   **Candidate Pool Consolidation:** After exiting the loop, if the threshold was never met, consolidate the candidate pool by taking the **union of all retrieved chunks** from all attempts. Deduplicate them based on their content/source.
    *   Sort the consolidated pool by their raw vector similarity scores, and cap the pool size at `candidate_k` (default 20) before sending to Step 3 (Rerank).

### **Phase 14 — Alternative Query Strategies (Multi-Query & HyDE)**
Implement Multi-Query Expansion and Hypothetical Document Embeddings.

1.  **Create prompts in `rag/prompts.py`:**
    *   Add `MULTI_QUERY_PROMPT`: Generates 3 distinct query variations.
    *   Add `HYDE_PROMPT`: Generates a short hypothetical answer passage.
2.  **Implement Multi-Query Strategy in `rag/query.py`:**
    *   ACT: Call LLM to generate `multi_query_n` (default 3) variants.
    *   ACT: Query Chroma with the original question + each generated variant.
    *   OBSERVE: Gather all chunks, union-deduplicate them, sort by vector similarity score, cap at `candidate_k`, and trace the result.
3.  **Implement HyDE Strategy in `rag/query.py`:**
    *   ACT: Call LLM to generate a hypothetical passage. Log the passage in the trace.
    *   ACT: Embed the passage and search Chroma for the top `candidate_k` chunks.
    *   OBSERVE: Log similarity scores and trace the retrieval.

### **Phase 15 — Local Reranker (P1) & Parameter Tuning (P1)**
Add finishing touches to make the app fully optimized and versatile.

1.  **Local Reranker Option (`reranker="local"`):**
    *   Use `sentence-transformers` cross-encoders (e.g. `BAAI/bge-reranker-base`).
    *   Download the model on first select, and run reranking locally on-device. This fulfills the fully-local, high-precision mode guarantee.
2.  **Trace Export & Download Button:**
    *   Add a button "Download Trace as JSON" on both the Ask and Trace tabs to make it easy for Builders to export trace data.
3.  **Gold-Set Expansion & Metric Evaluation:**
    *   Add 4 new hard/oblique questions to `eval/gold.jsonl` where Stage 1 failed due to bad initial similarity.
    *   Run evaluation and verify that `adaptive` strategy + Cohere/Local Reranking recovers these questions and achieves a higher hit@3 score.

---

## 4. Verification & Testing Plan

1.  **Grammar Validation:** Write an automated test script to parse trace logs and verify that every `ACT` has an preceding `REASON` and a succeeding `OBSERVE`.
2.  **Threshold Forced Evaluation:**
    *   Set `rewrite_trigger_score = 1.0` in sidebar. Submit a question, verify that exactly 2 rewrites occur in the trace before finishing.
    *   Set `rewrite_trigger_score = 0.0`. Verify no rewrites occur.
3.  **Resilience / Outage Drill:** Disable network or corrupt the `COHERE_API_KEY` while `reranker="cohere"` is selected. Verify that a warning appears in the UI and the system seamlessly falls back to vector order top-3.
4.  **Retrieval Quality Gold-Set Run:** Verify using `eval/hit_at_k.py` that the overall accuracy maintains or exceeds Stage 1 performance.
5.  **Sensitive Data Leak Check:** Confirm that neither `COHERE_API_KEY` nor other sensitive credentials ever enter any trace log files.
