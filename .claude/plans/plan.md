# Stage 3 — Evaluation Harness (Ragas) Implementation Plan

This document details the step-by-step implementation plan for **Stage 3: Evaluation Harness (Ragas)**. This system adds quantitative measurements to our RAG pipeline, making every future modification comparable against a committed baseline.

---

## 1. Technical Architecture Overview

The evaluation harness consists of:
1.  **A Golden Set (`evals/golden_set.json`):** A curated dataset of 10 questions with ground-truth reference answers and expected behaviors ("answer" or "refuse").
2.  **An Evaluation Runner (`evals/runner.py`):** A headless CLI and programmatic module that loads the golden set, executes the current retrieval/generation pipeline, runs LLM-as-a-judge assessments (using Ragas for factual queries and a custom judge for refusal queries), and manages rate limiting, concurrency, caching, and scorecard exports.
3.  **An Evaluate Tab in the UI:** A new tab in the Streamlit application that invokes the runner, streams progress question-by-question, displays an interactive scorecard table, provides full drill-down analysis, and renders deltas against a committed baseline.
4.  **A Weakest-Metric Panel:** An analytical dashboard that identifies the lowest aggregate metric, details its causes and remedies, and shows the specific questions that serve as reproduction cases.

---

## 2. Directory Layout Upgrades

We will introduce a new `evals/` root folder and update `app.py`, `requirements.txt`, and `rag/query.py`:

```
/home/coder/workspace/zero-to-rag/
├── app.py                      # Updated: Added "📊 Evaluate Pipeline" tab & scorecard views
├── requirements.txt            # Updated: Added pinned ragas, google-genai, langchain-google-genai
├── rag/
│   └── query.py                # Updated: Refactored ask_question return shape (Section 3 compliance)
└── evals/
    ├── README.md               # New: Documentation on CLI/UI execution and next steps
    ├── config.yaml             # New: Runner parameters, judge models, thresholds, cache folder
    ├── golden_set.json         # New: 10 curated questions across the 5 required categories
    ├── runner.py               # New: Evaluation executor, caching, concurrency, and Ragas integration
    ├── .cache/                 # New (gitignored): SHA-256 hashed metric result JSON files
    └── scorecards/             # New: Generated evaluation scorecards
        └── baseline.json       # New: Committed baseline scorecard (generated in step 6)
```

---

## 3. Step-by-Step Build Plan

### **Step 1: Prerequisite Pipeline Refactoring**
*   **Modify `ask_question` in `rag/query.py`:**
    *   Change the return structure to:
        ```python
        {
            "answer": str,
            "contexts": list[str],
            "meta": {
                "status": str,
                "retrieved_chunks": list[dict],
                "trace_id": str,
                "trace_events": list[dict]
            }
        }
        ```
    *   `contexts` will extract `chunk["content"]` in rank order.
    *   This ensures that callers requiring raw contexts (like the evaluation runner) can access them cleanly, while the existing `verify_tracing.py` and other sync caller scripts remain fully compliant.
    *   No changes are required in `app.py` for Chat View because `ask_question_stream` (which is what Chat View uses) is untouched.

### **Step 2: Golden Set Creation (`evals/golden_set.json`)**
*   Create a dataset of **10 questions** written strictly against the factual document corpus.
*   Category distribution:
    *   **4 `single_fact` questions** (2 on Acme Corp, 2 on Northstar Manufacturing)
    *   **2 `multi_hop` questions** (requiring cross-document or cross-section reading)
    *   **1 `aggregation` question** (combining values or lists from the text)
    *   **2 `out_of_scope` questions** (which must trigger refusal; expected behavior: `"refuse"`, no reference answer)
    *   **1 `ambiguous` question** (broad or with company-specific nuances; expected behavior: `"answer"`)
*   Schema matches Section 4 Task 1 exactly.

### **Step 3: Setup Dependencies and Config (`evals/config.yaml`)**
*   **Requirements:**
    *   Add exact pinned packages to `requirements.txt`:
        *   `ragas==0.2.14`
        *   `langchain-google-genai==1.0.10`
        *   `google-genai==0.1.1`
    *   Install dependencies inside the `.venv` virtual environment.
*   **Config file `evals/config.yaml`:**
    *   Add:
        *   `judge_model`: `"gemini-2.5-flash-lite"`
        *   `concurrency_limit`: 4
        *   `request_timeout`: 30
        *   `thresholds` (for delta alerts)
        *   `cache_dir`: `"evals/.cache"`
        *   `scorecard_dir`: `"evals/scorecards"`

### **Step 4: Build Headless Eval Runner (`evals/runner.py`)**
*   **Public API:**
    *   `run_eval(golden_set_path, config, on_result=None) -> Scorecard`
*   **Core Execution Steps:**
    *   Check for `GOOGLE_API_KEY` in environment. Fail fast if missing.
    *   Initialize LLM judge (`ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)`) wrapped in Ragas `LangchainLLMWrapper`.
    *   Run questions using an `asyncio.Semaphore` (bounded concurrency, default 4).
    *   For each question:
        *   Call synchronous query pipeline (`ask_question`).
        *   **Refusal Logic:** If `expected_behavior` is `"refuse"`, skip Ragas. Invoke the judge LLM to score `correct_refusal` ("Does this response decline to answer the question? Answer yes or no.").
        *   **Factual Logic:** Check the local metric cache under `evals/.cache/` first:
            *   Key is SHA-256 of `(question | contexts | answer | metric_name | judge_model | ragas_version)`.
            *   On cache hit, reuse the score.
            *   On cache miss, evaluate the specific metric via Ragas `evaluate`, then store the result.
        *   Handle rate limits (429) and server errors (5xx) with exponential backoff plus jitter.
        *   Handle per-question errors gracefully (save error info on the question row, don't crash).
        *   Fire `on_result` callback.
    *   **Aggregations:**
        *   Calculate the average for the 4 Ragas metrics (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) over non-refusal questions.
        *   Calculate `refusal_accuracy` separately (correct refusals / total refusal questions).
    *   **Scorecard Export:** Save scorecard JSON with comprehensive metadata to `evals/scorecards/<timestamp>_<shortsha>.json`.
    *   With `--save-baseline`, also write or overwrite `evals/scorecards/baseline.json`.

### **Step 5: Build Evaluate View (Tab 5 in `app.py`)**
*   Add a new tab: **`📊 Evaluate Pipeline`** alongside existing ones.
*   Include a high-contrast **"Run Evaluation"** button.
*   **Streaming progress:**
    *   Update progress bar and text status ("6/10 · 41s elapsed") using real-time callback updates.
    *   Append rows to a streaming dataframe as they complete.
*   **Scorecard Display:**
    *   **Header:** Show the 4 average metrics and refusal accuracy in high-visibility boxes using `st.metric`.
    *   **Delta Reporting:** If `baseline.json` exists, load it, calculate delta scores, and display them with color-coded up/down indicators.
    *   **Summary Table:** Detailed table of question-by-question metrics with the lowest/worst metric highlighted per row.
    *   **Drill-down Inspector:** An expandable accordion per question showing:
        *   Question and category.
        *   Generated answer vs. ground-truth reference.
        *   Sorted retrieved context chunks.
        *   Detailed scores and Ragas' per-claim faithfulness breakdown/reasons.
    *   **Actions:** Buttons to "Save scorecard as JSON" or "Promote this run as the baseline" (overwrites `baseline.json`).

### **Step 6: Weakest-Metric Callout Panel**
*   Below the scorecard, identify which of the 4 aggregate metrics has the lowest score.
*   Render a dedicated analytical card displaying:
    *   A warning alerting the user about the weakest metric.
    *   The exact table explaining causes and first levers (verbatim from Section 4 Task 5).
    *   The 2 lowest-scoring questions for that metric from the run, showing their IDs and questions to serve as concrete reproducer test cases.

### **Step 7: Baseline Generation and Documentation**
*   Wipe index and re-index the corpus using optimal defaults.
*   Run the evaluator headless CLI to generate and commit the initial `evals/scorecards/baseline.json`.
*   Author a thorough `evals/README.md` documenting CLI execution, UI integration, baseline promotions, and limitations.

---

## 4. Verification and Testing Plan

1.  **Strict Regression Test:** Check that existing Q&A streaming chat is fully functional and unchanged.
2.  **API Fallbacks and 429 Drill:** Ensure rate limits trigger exponential backoff.
3.  **Cache Efficiency:** Run the evaluation twice. Confirm the second run finishes in <10 seconds and makes exactly 0 judge calls (100% cache hit rate).
4.  **Refusal Accuracy:** Verify that out-of-scope questions are skipped by Ragas, evaluated by the custom judge, and averages are not diluted.
