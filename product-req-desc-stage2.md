# PRD — Stage 2: Adaptive Retrieval, Reranking & Decision Tracing

**Status:** Draft v1.0 · **Date:** 2026-07-21 · **Owner:** AI Solutions · **Sponsor:** CTO
**Builds on:** Stage 1 (`stage1-rag-prd.md`, implemented & verified). Ingestion side is unchanged; this PRD covers the query path and its observability.

---

## 1. Problem Statement

Stage 1's fixed pipeline (embed → top-3 → answer) fails silently on two fronts. First, oblique or badly-phrased questions retrieve weak candidates and the system has no way to recover — it answers from bad context or refuses. Second, and worse for an adaptive system: once the pipeline starts making decisions (rewrite or not, rerank or not), behavior changes per turn, and without a decision trace nobody — not the user, not the next debugger — can say *why* an answer came out the way it did. An adaptive system without a trace is a black box; a black box cannot be debugged, tuned, or trusted. The trace IS the harness for now.

## 2. Decisions This PRD Locks (review of the proposed architecture)

These resolve what the Stage 2 sketch leaves open, and flag its one governance trade-off:

1. **Cohere Rerank breaks the local-only guarantee — consciously.** The query plus up to 20 candidate chunks (document text) are sent to Cohere's API per reranked question. The UI shows a persistent "data leaves this machine" badge while the Cohere reranker is selected; the reranker enum includes `none` (pure local, Stage 1 behavior) and, at P1, a `local` cross-encoder so local-first mode survives Stage 2.
2. **Rerank never breaks a demo.** Missing/invalid `COHERE_API_KEY`, network failure, or rate limiting degrade gracefully to vector-order top-3, with the fallback recorded in the trace and surfaced in the UI — never a stack trace.
3. **Rewrite-loop exit semantics.** The score gate retries at most 2 rewrites; if the threshold is still unmet, the system proceeds anyway with the **union (deduplicated) of all candidate pools** from every attempt, capped at `candidate_k` by best vector score, then reranks. Low scores can change the path; they never fail the question — refusal remains the generation layer's job.
4. **Strategies are mutually exclusive at step 1** (per "instead of"): `adaptive` (plain embed + score-gated rewrite), `plain` (no rewrite — Stage 1 baseline), `multi_query`, `hyde`. The rewrite gate belongs to `adaptive` only; stacking transforms multiplies latency and makes traces unreadable.
5. **The score threshold is a config, not a constant.** 0.5 is the default gate on top vector similarity, but similarity distributions are corpus- and model-dependent; the slider plus the trace (which logs every gate check) is how the right value gets found per corpus.
6. **Trace-first build order.** Instrument the *existing* Stage 1 path before adding any new decision point — new branches must be born visible.
7. **Stage 1 stays reachable.** `strategy=plain` + `reranker=none` must reproduce Stage 1 ordering exactly — the permanent regression baseline and the control arm for every comparison.
8. **Conversation memory moves to Stage 3.** Earlier Stage 1 docs assumed memory was next; the actual Stage 2 is adaptive retrieval + observability. Statelessness (one question, one trace) is retained.

## 3. Goals

- **G1 — Recoverable retrieval:** questions whose initial top score falls below the gate get up to 2 LLM rewrites, measurably lifting candidate quality on oblique phrasings.
- **G2 — Precision at k:** cross-encoder reranking (20 → 3) yields final-context precision at least as good as vector order on the gold set, with a target uplift on hard questions.
- **G3 — Total decision visibility:** 100% of questions produce a complete REASON → ACT → OBSERVE timeline in the UI and a persisted JSONL trace on disk; a debugger can answer "why these 3 chunks?" from the trace alone.
- **G4 — User-swappable strategy:** retrieval strategy and reranker are selectable from the UI per session, taking effect on the next question, with the active combo recorded in every trace.
- **G5 — No silent egress:** any configuration that sends document content off-machine is visibly badged and individually consented to via key setup.

## 4. Non-Goals

- **Conversation memory / multi-turn** — Stage 3; keeps one-question-one-trace semantics clean.
- **Learned routing** (bandits, auto-tuned thresholds) — Stage 2's "adaptive" is rule-based and inspectable by design; a learned policy would reintroduce a black box before the harness exists to study it.
- **Agentic multi-hop retrieval / tool use** — out of scope; the graph has exactly the decision points in §6.
- **Formal eval automation** — the trace is the harness *for now*; a batch replay runner over the gold set is named as the P2 successor, not built here.
- **Ingestion changes** — same index, same embeddings; rerank and rewrites are strictly post-retrieval/query-side, so no re-indexing is required to adopt Stage 2.
- **Trace-UI sophistication** — no graphs, DAGs, or analytics dashboards; the trace view stays simple ordered rows ("don't over-engineer the trace UI"). The P2 replay runner, not the UI, is where analysis belongs.

## 5. User Stories

Personas: **Builder**, **Demo-er** (from Stage 1), plus **Debugger** — the person who did not write the code but must explain last Tuesday's answer.

- As a debugger, I want a REASON → ACT → OBSERVE timeline for every question, so that I can see each decision, its inputs, and its outcome. (P0)
- As a debugger, I want traces persisted to disk as JSONL with a trace ID shown in the UI, so that past sessions can be inspected after the fact. (P0)
- As a builder, I want to pick the retrieval strategy (adaptive / plain / multi-query / HyDE) and the reranker (Cohere / none) from the sidebar, so that I can compare behaviors on the same corpus. (P0)
- As a CTO, I want an unmissable indicator whenever a configuration sends document content off-machine, so that the local-first guarantee is traded knowingly, not silently. (P0)
- As a demo-er, I want rerank failures to fall back to vector order with a visible note, so that a live demo survives a missing key or dropped network. (P0)
- As a builder, I want per-chunk vector *and* rerank scores displayed side by side, so that I can see what the cross-encoder changed. (P0)
- As a builder, I want the gate threshold, max rewrites, and candidate pool size adjustable, so that the adaptive behavior can be tuned per corpus. (P1)
- As a builder, I want to download a question's trace as JSON, so that I can attach it to a bug report or diff two runs. (P1)

## 6. The Stage 2 Query Path (normative)

```
Question
  └─ Step 1  Strategy (user-selected):
       adaptive | plain ─ embed question → search Chroma (candidate_k = 20)
       multi_query ────── LLM makes 3 variants → search all (+ original) → union-dedupe → cap 20
       hyde ───────────── LLM writes hypothetical passage → embed it → search → top-20
  └─ Gate (adaptive only)  top vector score < rewrite_trigger_score (0.5)?
       yes → Step 2  LLM rewrites query → embed → search → recheck   (max 2 rewrites;
             then proceed with union-deduped pool regardless)
       no  → Step 3
  └─ Step 3  Rerank (user-selected): cohere (20 → top_k = 3) | none (vector top-3)
  └─ Step 4  Grounded prompt + 3 chunks → Ollama → streamed answer (+ Stage 1 refusal rule)
Throughout: every decision emits trace events → timeline in UI + JSONL on disk
```

## 7. Requirements

### Must-Have (P0)

**R1 — Decision trace (build first).** Every question produces an ordered trace of events, each `{trace_id, seq, timestamp, phase: REASON|ACT|OBSERVE, step, detail, duration_ms}`. Grammar: every ACT is preceded by a REASON (the decision and its inputs, e.g. "top score 0.42 < 0.50 → rewrite, attempt 1/2") and followed by an OBSERVE (what came back, e.g. "20 candidates, top score 0.63"). Traces render **visually as a timeline in a dedicated third Trace view** (Ingest · Ask · Trace) — session history, newest question expanded, past traces selectable by trace ID — with the latest question's timeline also inlined in the Ask view. Events append to `traces/YYYY-MM-DD.jsonl`. Scope guard: the timeline is deliberately simple — ordered, phase-tagged rows with durations and expandable payloads; no graph/DAG visualization (per the build note: don't over-engineer the trace UI).
- [ ] 100% of questions — including failures and fallbacks — produce a UI timeline and a JSONL record sharing the displayed trace ID.
- [ ] The UI has a third **Trace** view rendering the timeline visually; selecting a past trace ID re-renders that question's full timeline.
- [ ] Trace grammar holds: no ACT without its REASON and OBSERVE (validated in tests).
- [ ] Rewrites show before/after query text; gate checks show score vs. threshold; rerank shows candidate count in/out and both score sets; generation shows model, chunk sources, and answer length.
- [ ] Secrets (API keys) never appear in any trace or log.
- [ ] A tracing failure is itself logged but never blocks or alters the answer path.

**R2 — Score-gated rewrite loop (`adaptive` strategy, default).** After the initial search, if the top vector score < `rewrite_trigger_score`, the local LLM rewrites the query (retrieval-optimized rephrasing); embed → search → recheck; at most `max_rewrites = 2`; early-exit when the gate passes; on exhaustion proceed with the union-deduped candidate pool per §2.3.
- [ ] With threshold forced to 1.0, every question shows exactly 2 rewrites then proceeds; with 0.0, no question rewrites (both visible in trace).
- [ ] An oblique gold question that gates below threshold produces a trace showing the rewrite, and the final candidate pool includes chunks absent from the initial pool.
- [ ] The loop never errors out a question: an answer (or the Stage 1 refusal) is always produced.

**R3 — Cohere Rerank (cross-encoder), swappable.** With `reranker=cohere`, the query and up to `candidate_k` candidate chunk texts are sent to Cohere Rerank (`rerank_model`, default `rerank-v3.5`, configurable); the top `top_k = 3` by rerank relevance become the generation context. With `reranker=none`, vector-order top-3 is used (Stage 1 behavior).
- [ ] With a valid key: 20 in → 3 out; per-chunk display shows vector similarity and rerank score side by side; reordering vs. vector rank is visible.
- [ ] Missing/invalid key, network failure, or rate limit → visible warning, automatic fallback to vector top-3, and a trace REASON recording the fallback cause. No traceback.
- [ ] While `reranker=cohere` is selected, the UI shows a persistent "sends query + candidate chunks to Cohere" badge.
- [ ] `strategy=plain` + `reranker=none` reproduces Stage 1's exact ordering (regression test).
- [ ] **Quota guard:** Cohere is called at most once per question *by construction* — the rewrite loop's gate uses vector scores only, and rerank runs once after the loop concludes. A session call counter is visible in the sidebar and every call is a trace event (free-tier budget: 1,000 calls/month).

**R4 — Alternative query strategies (`multi_query`, `hyde`).** `multi_query`: one LLM call yields 3 variants; original + variants each searched at `candidate_k`; union-deduped by chunk ID; capped at `candidate_k` by best vector score. `hyde`: one LLM call writes a hypothetical answer passage; the passage's embedding is the search vector. Implementation may use LangChain's `MultiQueryRetriever` or a direct prompt — the behavior above is normative. *Priority note: the workshop outline marks these as stretch goals; they are P0 here per the sponsor directive that strategy swapping be user-selectable. Demote to P1 to match the outline if preferred — no other requirement depends on them.*
- [ ] Multi-query traces list every variant and per-variant top scores; the pool cap is enforced.
- [ ] HyDE traces include the full hypothetical passage.
- [ ] Both flow into Step 3 unchanged (rerank or fallback applies identically).

**R5 — Strategy & reranker selection in the UI.** Sidebar controls: retrieval strategy (`adaptive` default | `plain` | `multi_query` | `hyde`) and reranker (`cohere` default | `none`); changes apply to the next question; the active combo is stamped on the answer and into the trace.
- [ ] Switching either control mid-session changes the next question's path with no restart, confirmed by its trace.

**R6 — Preflight extension.** When `reranker=cohere`, preflight checks `COHERE_API_KEY` presence (env var / `.env` only — never `config.yaml`) and reports status alongside the Ollama checks, with fix guidance.
- [ ] Key absent while Cohere selected → sidebar shows the condition and the app still answers via fallback.

### Nice-to-Have (P1)

- **R7 — Tuning controls:** sliders for `rewrite_trigger_score` (0.0–1.0), `max_rewrites` (0–3), `candidate_k` (5–50); all values stamped into traces.
- **R8 — Trace export & retention:** per-question "Download trace (JSON)" button; keep the last `trace_keep = 200` questions on disk with rotation.
- **R9 — Local reranker option:** `reranker=local` using an on-device cross-encoder (e.g. a BGE reranker), restoring a fully local high-precision mode.
- **R10 — Gold-set additions:** +4 oblique/hard questions where Stage 1 `plain` fails the gate — the demonstration set for rewrite and rerank value.

### Future Considerations (P2)

- **Batch replay runner:** re-run the gold set through any strategy × reranker combo, aggregating hit@3 / MRR *from the trace files* — the trace schema is the eval substrate, which is why it's frozen and versioned from day one (`schema_version` field in every event).
- **Side-by-side compare view:** same question, two strategies, two timelines.
- Conversation memory (Stage 3 headline).

## 8. Configuration Spec (additions to Stage 1)

| Key | Default | Range / values | Takes effect | Notes |
|---|---|---|---|---|
| `retrieval_strategy` | `adaptive` | `adaptive` · `plain` · `multi_query` · `hyde` | next question | `plain` = Stage 1 baseline |
| `reranker` | `cohere` | `cohere` · `none` (P1: `local`) | next question | `none` = Stage 1 baseline |
| `rerank_model` | `rerank-v3.5` | any Cohere rerank model | next question | outline names the `rerank-v3.0` family — set this key to match; not hard-coded |
| `candidate_k` | 20 | 5–50 | next question | pool size sent to reranker |
| `top_k` | 3 | 1–10 | next question | final context size (Stage 1 key, new meaning: rerank output) |
| `rewrite_trigger_score` | 0.5 | 0.0–1.0 | next question | gate on top vector similarity; `adaptive` only |
| `max_rewrites` | 2 | 0–3 | next question | 0 disables the loop |
| `multi_query_n` | 3 | 2–5 | next question | variants per question |
| `query_llm_model` | = `llm_model` | any pulled Ollama model | next question | rewrites/variants/HyDE can use a smaller, faster model |
| `COHERE_API_KEY` | — | env var / `.env` only | startup | never stored in `config.yaml`, never logged |
| `trace_dir` | `./traces` | path | startup | JSONL, one file per day |
| `trace_keep` | 200 | 50–1000 | startup | retention (P1) |

## 9. Success Metrics

**Leading**
- **Trace completeness = 100%:** every question in a session yields a UI timeline + JSONL record; zero grammar violations in tests.
- **Debuggability check:** a person who has never seen the code answers "why these 3 chunks?" for 3 sample questions using only the trace view. Pass/fail, run once per release.
- **Precision:** hit@3 with `reranker=cohere` ≥ hit@3 with `reranker=none` on the full gold set (no regression), and ≥ +2 questions recovered on the hard/oblique subset (R10) by `adaptive` + rerank vs. `plain` + none.
- **Resilience:** the seeded failure drill (Cohere key removed mid-session) produces a fallback answer, a warning, and a trace entry — zero stack traces.
- **Latency visibility:** every trace shows per-step `duration_ms`; rerank step p50 ≤ 1.5 s; each LLM query-transform step's cost is individually attributable in the trace (absolute targets are hardware-dependent and tracked, not gated).

**Lagging**
- The P2 batch replay runner is buildable from persisted traces without schema changes (the schema held).
- ≥ 1 real retrieval bug or tuning decision is resolved *citing a trace* within the first month of use.

## 10. Timeline & Phasing

Hard dependency: Phase 10 (tracing) gates all other phases — new decision points must be born instrumented. Cohere account + API key needed before Phase 12.

| Phase | Scope | Exit criterion |
|---|---|---|
| **10 — Trace core** | Event schema (versioned), emitter, JSONL sink, UI timeline; instrument the existing Stage 1 path (retrieve → generate) | Stage 1 questions render complete REASON/ACT/OBSERVE timelines; grammar test green |
| **11 — Strategy & reranker selectors** | Sidebar enums, per-answer combo stamp, `plain`+`none` regression test | Stage 1 ordering reproduced under baseline combo |
| **12 — Cohere Rerank** | R3 incl. fallback, egress badge, preflight key check | R3 criteria pass; failure drill clean |
| **13 — Adaptive rewrite loop** | R2 gate, rewrite prompt, union-dedupe pool | R2 criteria pass; forced-threshold tests green; **Stage 1's failing question re-asked — gate fires, rewrite visible in the trace, answer recovers** |
| **14 — Multi-Query & HyDE** | R4 | R4 criteria pass |
| **15 — Tuning + gold set (P1)** | R7, R8, R10; measure §9 precision metrics | Comparison numbers recorded from traces |

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Document content sent to Cohere contradicts local-first expectations | Trust / governance | Explicit egress badge (R3); `none` and P1 `local` reranker options; key setup is an opt-in act |
| Cohere trial-key limits (1,000 calls/month total, 10 req/min on Rerank, non-commercial use only) or outage mid-demo | Demo failure, quota burn, licensing | ≤ 1 call per question by construction (never inside the rewrite loop); session call counter; automatic fallback to vector order with visible note + trace record; production/commercial use requires a paid key or the P1 local reranker |
| Threshold 0.5 miscalibrated for a given corpus | Useless or constant rewrites | Config slider (R7) + every gate check logged — tune from evidence, not guesses |
| Rewrite loop adds latency without lift | UX cost | Hard cap (2), early exit on gate pass, per-step durations in trace, `max_rewrites=0` kill switch |
| HyDE hypothetical drifts off-domain | Wrong-neighborhood retrieval | Passage fully visible in trace; strategy is user-swappable; rerank re-filters the pool |
| Trace files grow unbounded / leak sensitive text | Disk / privacy | Local-only storage, retention cap (R8), secrets excluded by construction (R1) |
| Rerank model name churn at Cohere | Breakage on their release cycle | `rerank_model` is config; preflight surfaces invalid-model errors with fix guidance |
| Baseline drift (Stage 2 refactor changes Stage 1 behavior) | Silent regression | Permanent `plain`+`none` equivalence test in CI (R3 last criterion) |

## 12. Open Questions

- **(CTO — blocking for Phase 12)** Is Cohere-by-default acceptable given the egress trade-off, or should `reranker=none` ship as default with Cohere as opt-in? This PRD assumes Cohere default per the Stage 2 brief; flipping it is a one-line config change.
- **(Engineering — non-blocking)** Same 3B model for rewrites/variants/HyDE, or a smaller faster model via `query_llm_model`? Trace durations from Phase 13 decide.
- **(Product — non-blocking)** Should the trace timeline also render for ingestion actions? Cheap once the emitter exists; deferred unless debugging demand appears.
- **(Engineering — blocking for P2 only)** Freeze `schema_version=1` field list now — proposed set is R1's; confirm before Phase 10 exit so the future replay runner never migrates.
