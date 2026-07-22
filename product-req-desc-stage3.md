
# Stage 3 — Evaluation harness (Ragas)

Implementation brief for a coding agent working in the existing RAG project repository. Read the whole document before writing code. Follow the order of work in section 8.

---

## 1. Context

The repo contains a working Stage 2 RAG agent: a document corpus, a retrieval + generation pipeline, and a UI with three existing views. Stage 3 adds measurement — an evaluation harness so that every future change to retrieval, chunking, prompts, or models produces a comparable score.

**Goal:** run a fixed golden set of questions through the agent, score each run with Ragas (LLM-as-judge), and display a scorecard in a new fourth UI view. Every change should become measurable against a committed baseline.

## 2. Before writing any code

1. Inspect the repo and identify: language/framework, UI framework, where the query pipeline is invoked, and how retrieved chunks flow through it.
2. Confirm the pipeline exposes retrieved contexts. The eval needs, per question: the answer **and** the list of retrieved chunk texts. If the current query function returns only an answer string, refactor it first (see section 3).
3. State your findings and plan in one short message before making large changes. If anything below conflicts with the repo's existing conventions, follow the repo's conventions and note the deviation.

## 3. Pipeline requirement (prerequisite refactor)

The core query function must return a structured result, e.g.:

```python
{"answer": str, "contexts": list[str], "meta": {...}}
```

- `contexts` is the list of chunk texts actually passed to the generator, in rank order.
- Update all call sites. Existing chat view behavior must be unchanged for the user.

## 4. Deliverables

### Task 1 — Golden set (`evals/golden_set.json`)

Plain JSON, committed to git. Ten questions authored **against the actual corpus** — read the documents and write reference answers from them; do not invent facts.

Schema:

```json
{
  "version": 1,
  "questions": [
    {
      "id": "q01",
      "category": "single_fact",
      "question": "…",
      "reference": "ground-truth answer written from the docs",
      "expected_behavior": "answer",
      "notes": "why this question is in the set"
    }
  ]
}
```

- `category` is one of: `single_fact`, `multi_hop`, `aggregation`, `out_of_scope`, `ambiguous`.
- Distribution for the 10 questions: 4 single_fact, 2 multi_hop, 1 aggregation, 2 out_of_scope, 1 ambiguous.
- `expected_behavior` is `"answer"` or `"refuse"`. The 2 out_of_scope questions use `"refuse"` and have no `reference` answer — the correct behavior is declining because the corpus does not cover them.

### Task 2 — Eval runner (`evals/runner.py`)

A callable module with a CLI entry point. **No UI imports** — the UI wraps this module; a future CI job will call it headlessly.

- Public API: `run_eval(golden_set_path, config, on_result=None) -> Scorecard`, where `on_result` is an optional callback fired after each question completes (the UI uses this for streaming). CLI: `python -m evals.runner [--save-baseline]`.
- Metrics via Ragas: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`. Note that recall and precision require the `reference` field.
- Judge model: **Gemini Flash Lite** (free tier), API key from the `GOOGLE_API_KEY` env var only — never hardcoded. Fail fast with a clear message if the key is missing.
- Config file `evals/config.yaml`: judge model id, concurrency limit, request timeout, thresholds, cache directory, scorecard output directory.
- **Pin versions.** Add exact versions of `ragas` and the Gemini client/wrapper libraries to the dependency lockfile. Ragas' API surface changes between versions — consult the docs for the pinned version rather than coding from memory (dataset construction, metric names, and LLM wrapper interfaces all vary).
- **Refusal questions:** skip the four Ragas metrics for `expected_behavior: "refuse"`. Instead score a boolean `correct_refusal` with one judge call ("Does this response decline to answer the question? Answer yes or no."). Report refusal accuracy separately; never include refusal questions in the four-metric averages.
- **Concurrency and rate limits:** run judge calls concurrently with a bounded semaphore (default 4, configurable). On HTTP 429 or 5xx, retry with exponential backoff plus jitter. The free tier's rate limit, not model speed, is the practical ceiling — a full run is ~40+ judge calls and ~90 seconds is acceptable.
- **Cache:** key = SHA-256 of `(question | contexts | answer | metric_name | judge_model | ragas_version)`. Store results as JSON under `evals/.cache/` (gitignored). On a cache hit, skip the judge call entirely. A rerun with no pipeline changes must make zero judge calls.
- **Scorecard output:** JSON containing per-question scores, per-metric aggregates, refusal accuracy, and run metadata — timestamp, git commit hash, ragas version, judge model, wall time, judge-call count, cache-hit count. Write to `evals/scorecards/<timestamp>_<shortsha>.json`. With `--save-baseline`, also write `evals/scorecards/baseline.json` (committed to git).

### Task 3 — Evaluate view (fourth UI view)

- Add alongside the three existing views, matching their styling and navigation pattern.
- A "Run evaluation" button starts a run. **Stream results:** each question's row appears or updates the moment it finishes, with a progress indicator ("6/10 · 41s elapsed"). Never a single blocking spinner for the whole run.
- Per-question errors (judge failure, rate-limit stall) are shown on that row without aborting the rest of the run.

### Task 4 — Scorecard display

- Header: the four aggregate metrics (0–1, two decimals) plus refusal accuracy. When `baseline.json` exists, show the delta per metric with up/down indicators.
- Table: one row per question — id, category, four scores (or refusal result), with each row's worst metric visually highlighted.
- Drill-down per question: the question, retrieved chunks, the agent's answer, the reference answer, per-metric scores, and Ragas' per-claim faithfulness breakdown.
- Actions: save this scorecard; set this run as the new baseline.

### Task 5 — Weakest-metric callout

A panel below the scorecard showing the lowest aggregate metric, its likely causes and first levers (embed the table below verbatim), and the two lowest-scoring questions as repro cases.

| Metric down | Likely cause | First levers |
|---|---|---|
| Context recall | Right info never made top-K | Chunk size/overlap, embedding model, raise K, query rewriting |
| Context precision | Noise crowding signal | Add a reranker, lower K, metadata filters |
| Faithfulness | Model inventing beyond chunks | Grounding instructions, require citations, lower temperature, model swap |
| Answer relevance | Correct but off-target | Prompt structure, question restatement, answer-format constraints |

## 5. Non-functional requirements

- Secrets via env vars only; nothing sensitive in git.
- Full 10-question run completes in ≤ ~2 minutes on the free tier; a fully cached rerun completes in ≤ 10 seconds.
- The runner module works headlessly (CLI) with identical results to the UI path.
- Add `evals/README.md`: how to run the eval (UI and CLI), how to add a question, how to set the baseline, and known limitations. List "CI gating" and "growing the set to 50+" as next steps.

## 6. Acceptance criteria

- [ ] `evals/golden_set.json` exists with 10 questions across the 5 categories, including 2 refusal cases, and is committed
- [ ] `python -m evals.runner` completes and writes a scorecard JSON with full run metadata
- [ ] An immediate rerun with no changes makes zero judge calls (cache hit)
- [ ] The Evaluate view streams per-question results with a progress indicator; no blocking full-run spinner
- [ ] Drill-down shows retrieved chunks, answer, reference, and per-claim faithfulness
- [ ] `--save-baseline` writes `baseline.json`; the UI renders deltas against it
- [ ] `ragas` and judge client library versions are pinned in the lockfile
- [ ] Refusal questions are scored via `correct_refusal` and excluded from the four-metric averages
- [ ] A 429 during a run triggers backoff and the run still completes
- [ ] `evals/README.md` is written
- [ ] Existing chat functionality is unchanged after the pipeline refactor

## 7. Out of scope — do not build now

- CI integration or merge gating (next phase; the headless runner is the only prerequisite, and it is in scope)
- Growing the golden set beyond 10 questions
- Trend dashboards or historical charts across runs
- Human-labeling or judge-calibration tooling
- Alternative judge models or judge ensembles

## 8. Order of work

1. Inspect repo, report stack and plan, refactor the pipeline return shape (section 3).
2. Author the golden set from the corpus (Task 1).
3. Build the runner: Ragas wiring, refusal handling, concurrency, cache, scorecard output, CLI (Task 2).
4. Build the Evaluate view and scorecard display (Tasks 3–4), wiring the streaming callback.
5. Add the weakest-metric callout (Task 5).
6. Run the full set once, save it as baseline, commit golden set + baseline + README, and verify every acceptance criterion.

## 9. Questions to resolve by inspecting the repo (do not ask the user unless blocked)

- Exact UI framework and view-registration pattern for adding the fourth view.
- Where the query pipeline lives and every call site affected by the section 3 refactor.
- Whether an async runtime already exists (reuse it for concurrency) or a thread pool is more idiomatic.
- The correct Ragas dataset-construction and LLM-wrapper APIs **for the pinned version** — verify against its documentation before use.

