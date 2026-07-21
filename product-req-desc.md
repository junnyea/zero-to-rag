
# Product Design Document — Stage 1: Classic RAG over Your Docs

**Working name:** Local Doc Q&A (LangChain + Chroma + Ollama)
**Status:** Draft v1.0 · **Date:** 2026-07-21 · **Owner:** AI Solutions

---

## 1. Summary

A fully local, single-user RAG application. The user uploads documents (TXT / MD / PDF), the app chunks and embeds them into a persistent Chroma vector store, and the user asks single-turn questions that a local Ollama model answers grounded in the top-3 retrieved chunks. No API keys, no data leaves the machine. Chunk size (and overlap) are first-class, user-tunable parameters so the effect of chunking on retrieval quality is *observable and measurable* — that is the core learning objective of Stage 1.

---

## 2. Design Review of the Original Outline

An evaluation of the five-step outline: what it gets right, and what this document changes to get the best out of it.

### Strong as-is
- **Minimal, correct architecture.** Two pipelines (ingest, query) sharing one vector store is the canonical RAG shape — nothing extraneous.
- **Local-first stack.** Ollama + Chroma means zero cost, zero keys, and privacy by default; ideal for demos with real internal docs.
- **Chunk size as config.** Chunking is the single highest-leverage RAG parameter; exposing it turns the app into a teaching instrument, not just a demo.
- **Top-3 retrieval.** Keeps prompt context small enough for small local models to stay coherent and fast.

### Gaps closed in this design
1. **Model pulls moved out of runtime.** The outline's footnote warns the first `nomic-embed-text` call pulls ~270 MB — a hazard mid-session. Fix: an explicit Step 00 (`ollama pull ...` before first run) plus an in-app preflight check. The app never auto-pulls, so "don't restart mid-pull" can't bite during a demo.
2. **Chunk overlap exposed next to chunk size.** Size without overlap gives misleading experiments because boundary effects dominate; the two are tuned together.
3. **Retrieval transparency.** The Ask view shows the retrieved chunks, similarity scores, and source filenames. Without this, you cannot see *why* an answer is good or bad, and the chunk-size experiment has no visible effect.
4. **Grounding and refusal behavior.** The prompt instructs the model to answer only from the provided context, cite sources, and say "I can't find that in the ingested documents" otherwise. This reduces hallucination and makes failures diagnosable.
5. **Persistence and re-ingest semantics defined.** Chroma persists to disk (no re-ingesting every restart); re-uploading a file *replaces* its chunks via hash-based IDs instead of duplicating them; a "Reset index" control exists.
6. **Friendly failure handling for Ollama.** The two failures the outline's footnote predicts (server not running, model not pulled) surface as actionable banners with copy-paste fix commands — never stack traces.
7. **A measurable definition of "best".** A small gold Q&A set with a hit@3 script, so chunk-size changes are evaluated objectively rather than eyeballed.
8. **"No memory" made an explicit boundary.** Single-turn Q&A is a stated non-goal (conversation memory is Stage 2's headline feature), so nobody wires up chat history by accident and the Stage 1 → Stage 2 delta stays teachable.

---

## 3. Problem Statement

Teams want to ask questions of private documents without sending content to external APIs, and builders learning RAG lack a minimal reference implementation where the key levers (chunking, retrieval depth, grounding) are visible and tunable. Without one, RAG behavior stays a black box: chunking parameters get cargo-culted, and nobody can explain why retrieval succeeded or failed. The cost is slow skill-building and low-trust demos.

---

## 4. Goals

- **G1 — Fast time-to-value:** clean clone → first grounded answer in ≤ 15 minutes, including model pulls.
- **G2 — Fully local:** works offline after the initial pulls; zero external API calls at runtime.
- **G3 — Explorable chunking:** changing chunk size/overlap re-indexes the corpus and produces a measurable change in retrieval (visible in the UI and in the hit@3 script).
- **G4 — Grounded answers:** responses cite source files; out-of-corpus questions get an explicit refusal, not a fabrication.
- **G5 — Readable reference:** core logic ≤ ~500 LOC excluding UI boilerplate, so the whole system fits in one sitting.

---

## 5. Non-Goals

- **Conversation memory / multi-turn chat.** Stage 2's headline feature; keeping Stage 1 stateless keeps the data flow linear and inspectable.
- **Multi-user, auth, or deployment.** This is a single-user localhost tool.
- **Advanced retrieval** (hybrid/BM25, reranking, query rewriting). Would obscure the classic-RAG baseline this stage exists to teach.
- **Document management UI** beyond replace-on-reupload and full reset. Per-file delete/rename adds surface area without teaching value.
- **Generation-quality evaluation.** Stage 1 evaluates *retrieval* (model-free hit@k); judging LLM answer quality is a later concern.

---

## 6. Users & User Stories

Personas: **Builder** (engineer/PM learning RAG hands-on) and **Demo-er** (solutions lead showing local RAG to a stakeholder).

- As a builder, I want to upload a handful of docs and ask questions, so that I see end-to-end RAG working locally. (P0)
- As a builder, I want to change chunk size/overlap and re-index, so that I can observe the effect on retrieval quality. (P0)
- As a builder, I want to see which chunks were retrieved (with scores and sources), so that I can debug bad answers. (P0)
- As a demo-er, I want clear guidance when Ollama isn't running or a model isn't pulled, so that a live demo doesn't die on a traceback. (P0)
- As a builder, I want the index to persist between runs, so that I don't re-ingest on every restart. (P0)
- As a builder, I want a scripted hit@3 eval over a small gold set, so that chunking changes are compared objectively. (P1)
- As a demo-er, I want streamed token output, so that the local model feels responsive. (P1)

---

## 7. System Overview

```
INGEST  files (.txt/.md/.pdf)
         → loader
         → RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
         → OllamaEmbeddings("nomic-embed-text")        [batched]
         → Chroma PersistentClient  ./chroma_db  (cosine)

ASK     question
         → embed (same model)
         → Chroma similarity search, top_k = 3
         → grounded prompt template (context + question)
         → ChatOllama("llama3.2:3b")
         → answer + [source] citations + retrieved-context panel
```

| Component | Choice | Notes |
|---|---|---|
| Orchestration | LangChain (`langchain`, `langchain-ollama`, `langchain-chroma`, community loaders) | Used thinly — loaders, splitter, embeddings, chat model. No opaque chain magic. |
| Embeddings | `nomic-embed-text` via Ollama | 768-dim, ~270 MB single pull, strong quality at this size. |
| LLM | `llama3.2:3b` via Ollama (configurable) | Runs on 8 GB RAM laptops; swap to an 8B model via config when hardware allows. |
| Vector store | Chroma `PersistentClient`, collection `docs`, cosine distance, telemetry off | Persistence is what makes the chunk-size experiment loop practical. |
| UI | Streamlit — two tabs (Ingest / Ask) + config sidebar | See ADR-1. |
| Config | `config.yaml` at startup, editable live in the sidebar | See Section 10. |

---

## 8. Functional Requirements

### 8.1 Ingestion pipeline (P0)

- Accept `.txt`, `.md`, `.pdf` via multi-file uploader.
- Split with `RecursiveCharacterTextSplitter` using the configured `chunk_size` / `chunk_overlap`.
- Attach metadata per chunk: source filename, chunk index, character span.
- **Replace semantics:** chunk IDs are `sha256(file_bytes)[:12] + ":" + chunk_index`; ingesting a file first deletes any existing chunks for that filename, then adds the new ones. Re-upload never duplicates.
- Embed in batches (e.g., 32) with a progress indicator; report chunks written per file.
- Record `embed_model`, `chunk_size`, and `chunk_overlap` in collection metadata. If the configured embed model later differs from the recorded one, block ingestion/query and require "Reset & re-index" (prevents silent embedding-dimension mismatch).

**Acceptance criteria**
- [ ] A 20-page PDF ingests in ≤ 60 s on a modern laptop and reports its chunk count.
- [ ] Re-uploading the same file leaves total chunk count unchanged (replace, not append).
- [ ] Changing chunk size and pressing "Re-index all" rebuilds the collection; the new chunk count reflects the new size.
- [ ] Given Ollama is down, when the user ingests, then a friendly banner with the exact `ollama serve` command appears — no traceback.

### 8.2 Query pipeline (P0)

- **Single-turn:** every question is independent; no history is stored or injected (see ADR-5).
- Embed the question with the same embedding model; similarity search with `top_k` (default 3).
- Grounded prompt template: answer **only** from the provided context; cite sources inline as `[filename]`; if the context doesn't contain the answer, reply exactly: *"I can't find that in the ingested documents."*
- Response view: the answer, a sources line, and an expandable "Retrieved context" panel showing each chunk with its similarity score and source file.
- P1: stream tokens as they generate.

**Acceptance criteria**
- [ ] A question answerable from the corpus yields an answer containing ≥ 1 `[source]` citation matching a real ingested file.
- [ ] An out-of-corpus question (e.g., world trivia vs. an HR handbook) yields the refusal string — no fabricated answer.
- [ ] The retrieved panel shows exactly `top_k` chunks, sorted by score descending.
- [ ] With an empty index, the Ask view prompts the user to ingest first instead of erroring.

### 8.3 UI (P0)

- **Ingest tab:** file uploader → "Ingest" button → per-file results; index stats card (files, chunks, embed model, chunk params); "Re-index all" and "Reset index" (with confirmation).
- **Ask tab:** question input → "Ask" → answer, sources, retrieved-context expander.
- **Sidebar:** config controls (see Section 10) and a preflight status block — Ollama server ●, embed model ●, chat model ● — each red state paired with its copy-paste fix command.

### 8.4 Preflight & operations (P0)

- On startup: ping `http://localhost:11434`; verify both models appear in the local model list; render status in the sidebar.
- README defines **Step 00** (see Section 14): pull both models and start `ollama serve` in a separate terminal *before* the app runs — this removes the ~270 MB mid-session pull entirely.
- The app never auto-pulls models. A failed preflight blocks Ingest/Ask actions with instructions rather than letting calls fail deep in the stack.

### 8.5 Evaluation harness (P1)

- `eval/gold.jsonl`: ~8 `{question, expected_source}` pairs authored against the sample corpus.
- `python eval/hit_at_k.py --k 3`: prints **hit@k** — the fraction of gold questions whose expected source file appears in the top-k retrieved chunks. Model-free, so it isolates retrieval quality from generation quality.
- Powers the chunk-size experiment in Section 13.

### 8.6 Future-proofing (P2 — design for, don't build)

- Retrieval sits behind one small function so hybrid search or a reranker can slot in for Stage 2 without touching the UI.
- The prompt template lives in one file for later citation-format upgrades.
- Collection metadata already versions index parameters, enabling future side-by-side index comparisons.

---

## 9. Key Design Decisions (ADR-lite)

### ADR-1 — UI framework: **Streamlit**

| Option | Effort | File upload / tabs | Fit for stateless Q&A | Notes |
|---|---|---|---|---|
| **Streamlit ✓** | Low | Native | Excellent | Pure Python, no build step, sidebar config for free. |
| Gradio | Low | Native | Good | Close second; chat-styled components nudge toward multi-turn, which Stage 1 deliberately avoids. |
| React | High | Custom | Good | Best UX ceiling, but needs a FastAPI backend and a build chain — ~10× effort for zero Stage 1 learning value. |
| Plain HTML | Medium | Custom | Poor | Still requires a Python backend; worst of both worlds here. |

**Consequence:** Python-only repo, fastest path to the two-view spec. Revisit only if a later stage needs a productized frontend.

### ADR-2 — Chunking: **RecursiveCharacterTextSplitter, character-based, default 800 / 150**

Token-based splitting couples the pipeline to a tokenizer; semantic chunking hides the very lever this stage teaches. Character-based splitting is dependency-free and predictable. Default `chunk_size=800` (~200 tokens) with `chunk_overlap=150` suits handbook/policy prose and a 768-dim embedder; both are sliders, and the eval harness — not intuition — decides the final default.

### ADR-3 — Models: **`nomic-embed-text` + `llama3.2:3b`**

- Embeddings: strong retrieval quality for a single ~270 MB pull; 768 dims keeps the index small. **Quality note:** nomic-embed-text was trained with task prefixes — prepend `search_document: ` at ingest and `search_query: ` at query time (LangChain's wrapper does not add these). Apply consistently on both sides; validate the lift with the eval harness.
- LLM: a 3B default keeps the floor at ~8 GB RAM machines. `llm_model` is config, so 16 GB+ machines can switch to an 8B (e.g., `llama3.1:8b`, `qwen2.5:7b`) with no code change.
- **Consequence:** a small model means tight context discipline — hence top-3 default and a strict grounding prompt.

### ADR-4 — Vector store: **Chroma, persistent on disk** (not in-memory)

The naive in-memory default forces full re-ingestion on every restart, which kills the tune-chunks → re-index → re-ask loop. Persistent client at `./chroma_db`, cosine distance (`hnsw:space=cosine`), anonymized telemetry off. **Consequence:** replace/reset semantics must be explicit — defined in 8.1.

### ADR-5 — Statelessness: **single-turn only** (the outline's "no memory")

Each query is independent. This is a feature, not a limitation: the data flow stays linear (question → retrieve → answer), failures are attributable, and conversation memory arrives in Stage 2 as a clean, teachable delta.

---

## 10. Configuration Spec

| Key | Default | Range / values | Takes effect |
|---|---|---|---|
| `chunk_size` | 800 | 200–2000 (step 100) | Next ingest / re-index |
| `chunk_overlap` | 150 | 0–400, must be < chunk_size / 2 | Next ingest / re-index |
| `top_k` | 3 | 1–10 | Next question |
| `embed_model` | `nomic-embed-text` | Any pulled Ollama embedding model | **Requires Reset & re-index** |
| `llm_model` | `llama3.2:3b` | Any pulled Ollama chat model | Next question |
| `persist_dir` | `./chroma_db` | Path | Startup |

Rule: retrieval-time parameters (`top_k`, `llm_model`) apply instantly; index-time parameters (`chunk_*`, `embed_model`) mark the index **stale** in the UI and surface a "Re-index" call-to-action rather than silently mixing regimes.

---

## 11. Error Handling & Operational Notes

| Failure | App behavior |
|---|---|
| Ollama server not running | Red preflight banner + copyable `ollama serve` command; Ingest/Ask disabled. Run `ollama serve` in a **separate terminal before** launching the app. |
| Model not pulled | Banner + exact `ollama pull nomic-embed-text` / `ollama pull llama3.2:3b` command. The app never auto-pulls — a ~270 MB download mid-demo (or a restart during the pull) is the known failure mode this design removes. |
| Embed model ≠ collection metadata | Hard stop with "Reset & re-index" action. Prevents silent dimension mismatch producing nonsense similarity scores. |
| PDF extracts no text (scanned) | Per-file warning, file skipped. OCR is out of scope. |
| File > 10 MB | Rejected with message — keeps demo latency predictable. |
| Empty index on Ask | Inline prompt to ingest first. |

---

## 12. Evaluation Plan — the "get the best out of it" loop

1. **Sample corpus** (outline Step 01): generated HR handbook, two policy docs, and an FAQ, committed to `sample_docs/`.
2. **Gold set:** 8 questions with known `expected_source` (and ideally the answer phrase) in `eval/gold.jsonl`.
3. **Metric:** hit@3 — correct source file present in the retrieved set. Model-free, cheap, isolates retrieval.
4. **Chunking experiment:** run hit@3 across `chunk_size ∈ {300, 800, 1500}` × `overlap ∈ {0, 150}`; record hit@3 and mean retrieved-chunk length; promote the winner to the default config.
5. **Qualitative demo script:** three canned questions, including one guaranteed out-of-corpus miss to showcase the refusal behavior — the moment that builds stakeholder trust.

---

## 13. Success Metrics

**Leading**
- Clean clone → first grounded answer in ≤ 15 minutes on a fresh machine (timed).
- hit@3 ≥ 0.8 on the gold set at default config.
- Zero user-facing stack traces across the two seeded failure drills (Ollama down; model missing).
- Changing chunk size 800 → 300 visibly changes retrieved chunks for ≥ 1 demo question.

**Lagging (workshop / enablement context)**
- ≥ 80% of participants complete Stage 1 end-to-end.
- Stage 2 builds on this repo without interface changes (i.e., the seams in 8.6 held).

---

## 14. Build Plan (maps to the outline, plus the missing Step 00)

| Step | Scope | Done when |
|---|---|---|
| **00 — Setup** *(new)* | Install Ollama; `ollama pull nomic-embed-text`; `ollama pull llama3.2:3b`; start `ollama serve` in its own terminal; `pip install -r requirements.txt`. | Preflight shows all green. |
| **01 — Sample corpus** | Generate handbook / policies / FAQ with Claude Code → `sample_docs/`. | 4+ files, plain text or MD (PDF optional). |
| **02 — Scaffold** | Repo layout below; pinned `requirements.txt`; `config.yaml` with defaults from Section 10. | App boots to empty two-tab UI with preflight sidebar. |
| **03 — Ingestion** | Section 8.1. | 8.1 acceptance criteria pass. |
| **04 — Query** | Section 8.2. | 8.2 acceptance criteria pass. |
| **05 — UI polish** | Section 8.3 complete; stale-index indicator; reset confirm. | Demo script runs clean. |
| **06 — Eval (P1)** | Section 8.5 harness; run the chunking experiment; update defaults. | hit@3 report committed. |

```
app.py                  # Streamlit entry: tabs, sidebar, preflight
rag/
  config.py             # load/validate config.yaml
  preflight.py          # Ollama + model checks
  ingest.py             # load → split → embed → upsert (replace semantics)
  query.py              # embed → retrieve → grounded prompt → answer
  prompts.py            # the one grounded QA template
eval/
  gold.jsonl
  hit_at_k.py
sample_docs/
config.yaml
requirements.txt        # pinned versions
chroma_db/              # gitignored
```

---

## 15. Open Questions

- **(Owner)** What is the weakest target machine? Confirms whether `llama3.2:3b` is the right floor or the default can move to an 8B. *Non-blocking — config default.*
- **(Engineering)** Apply the nomic task prefixes (ADR-3) in v1, or land the eval harness first and let hit@3 prove the lift? *Non-blocking — harness decides.*
- **(Product)** Is per-file delete needed in Stage 1, or is replace-on-reupload + full reset sufficient? *Blocking only for UI scope in Step 05.*

---

## 16. Stage 2 Preview (parking lot — explicitly not now)

Conversation memory (the headline delta), citation-aware answer formatting, hybrid retrieval (BM25 + vector), reranking, query rewriting, per-file management, multiple collections/corpora, and answer-quality evaluation to complement retrieval hit@k.

