# Stage 3 — Evaluation Harness (Ragas & LLM-as-a-Judge)

The Stage 3 Evaluation Harness introduces quantitative measurements to the RAG repository. It runs a committed golden set of 10 questions, scores them using **Ragas (v0.4.3)** and custom LLM-as-a-judge classifiers via **Google Gemini Flash Lite**, and displays a comprehensive scorecard comparing results against a baseline.

---

## 1. Directory Layout

All evaluation assets are self-contained in the `evals/` directory:

```
evals/
├── README.md               # This documentation file
├── config.yaml             # Runner configuration (model, concurrency, cache, and output folders)
├── golden_set.json         # Standardized 10-question evaluation dataset written from real corpus
├── runner.py               # Evaluation engine, caching, concurrency, and Ragas integrations
├── .cache/                 # Gitignored folder containing metric-level evaluation cache
└── scorecards/             # Output folder for generated scorecard JSONs
    └── baseline.json       # Committed baseline scorecard to compare performance changes
```

---

## 2. Core Concepts

### A. Golden Set Schema (`evals/golden_set.json`)
The evaluation uses a fixed 10-question golden set designed against the document corpus (`sample_docs/`), distributing queries across five categories to stress-test specific capabilities of the RAG pipeline:
*   **`single_fact` (4 questions):** Direct fact retrieval (e.g. office locations, policy limits).
*   **`multi_hop` (2 questions):** Queries requiring information combined from multiple document sections.
*   **`aggregation` (1 question):** Combining numbers or lists from multiple sections (e.g. division revenue).
*   **`out_of_scope` (2 questions):** Irrelevant queries (e.g. maternity leave) where the expected behavior is **refusing** to answer.
*   **`ambiguous` (1 question):** Broad or company-specific nuanced questions.

### B. Evaluation Metrics
*   **Faithfulness (0-1):** Measures whether the generated answer is strictly grounded in and supported by the retrieved contexts.
*   **Answer Relevancy (0-1):** Measures how directly the generated answer addresses the user's initial question.
*   **Context Precision (0-1):** Evaluates if the retriever correctly ranks relevant chunks higher than irrelevant ones.
*   **Context Recall (0-1):** Assesses if the retrieved contexts cover all the information needed to answer the question, using the reference answer as the proxy.
*   **Refusal Accuracy (0-1):** Refusal questions are automatically excluded from the four Ragas metrics above. Instead, they are classified via a single judge call to score `correct_refusal` (yes/no) to measure refusal correctness.

### C. Metric-Level SHA-256 Caching
To prevent redundant API calls to Google's Gemini Flash Lite and maintain a high rerun speed, the runner caches individual scores. The cache key is calculated as:
$$\text{Key} = \text{SHA-256}(\text{question} \mid \text{contexts} \mid \text{answer} \mid \text{metric\_name} \mid \text{judge\_model} \mid \text{ragas\_version})$$
Scores are saved in `evals/.cache/`. On reruns with no pipeline changes, the system achieves a **100% cache hit rate**, completing the evaluation run in **under 10 seconds** with **exactly 0 judge calls**!

---

## 3. How to Run Evaluations

### Prerequisites
Make sure your Google Gemini API Key is loaded in your environment variables:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

Ensure your Ollama server is running locally with the embedding and LLM models loaded, and that the Python virtual environment is activated:
```bash
source .venv/bin/activate
```

---

### Run via Streamlit UI (Recommended)
1.  Launch the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2.  Navigate to the fifth tab: **`📊 Evaluate Pipeline`**.
3.  Click **"🚀 Run Complete Evaluation"**.
4.  The system will stream results in real-time, displaying a question-by-question progress bar and updating the scores table.
5.  **Actions available in the UI:**
    *   **Drill Down:** Click on any question accordion to see the full Question, actual answer, reference answer, sorted retrieved context chunks, and detailed scores.
    *   **Save Baseline:** Click **"Save as New Baseline"** to promote the current run's scores as the new default baseline (`evals/scorecards/baseline.json`).
    *   **Download Scorecard:** Export the full scorecard JSON directly to your local computer.

---

### Run via Headless CLI
You can execute evaluations headlessly from your terminal, which is ideal for automation or CI/CD pipelines.

To run and write a timestamped scorecard file:
```bash
python3 -m evals.runner
```

To run and immediately overwrite the committed baseline:
```bash
python3 -m evals.runner --save-baseline
```

---

## 4. How to Add a Question to the Golden Set
1.  Open `evals/golden_set.json`.
2.  Append a new question dictionary to the `questions` array, conforming to the schema:
    ```json
    {
      "id": "q11",
      "category": "single_fact",
      "question": "Your question here?",
      "reference": "Expected ground-truth answer written from the documents.",
      "expected_behavior": "answer",
      "notes": "Description of why this question is in the evaluation set."
    }
    ```
    *Note:* For out-of-scope questions, set `expected_behavior` to `"refuse"` and `reference` to `null`.
3.  Commit the updated `golden_set.json` to Git.

---

## 5. Next Steps & Future Enhancements
*   **CI Gating:** Integrate the headless runner (`python3 -m evals.runner`) as an automated step in your CI/CD workflow (GitHub Actions, GitLab CI). If any metric drops below the defined thresholds in `evals/config.yaml`, the build or PR merge can be blocked.
*   **Growing the Golden Set to 50+:** Scale up the golden set to cover larger corporate domains and various linguistic variations to achieve more robust, statistically significant, and high-coverage evaluation scores.

---

## 6. Known Limitations
*   **Rate Limits:** The Gemini free tier has rate limits. Bounded concurrency (semaphore of 4) is implemented with exponential backoff plus jitter to prevent rate exhaustion, but very large golden sets (50+ questions) may require upgrading to a paid Tier.
*   **Ollama Speed:** Running locally on CPU can cause the query pipeline step to bottleneck. Ensure your Ollama server has hardware acceleration (GPU) configured for optimal execution.
