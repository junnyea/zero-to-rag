import os
import sys
import json
import time
import yaml
import hashlib
import asyncio
import argparse
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional, Tuple

import git
import ragas
from ragas import evaluate, EvaluationDataset
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from openai import OpenAI
from google import genai

# Add project root to path just in case
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import load_config as load_rag_config
from rag.query import ask_question

# Ensure cache directory is created
CACHE_DIR = "evals/.cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------- Helper Functions -----------------

def get_git_commit() -> str:
    """Returns the short Git commit SHA of the current repository."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:7]
    except Exception:
        return "unknown"

def get_cache_key(
    question: str,
    contexts: List[str],
    answer: str,
    metric_name: str,
    judge_model: str,
    ragas_version: str
) -> str:
    """Generates a stable SHA-256 cache key based on evaluation parameters."""
    norm_q = (question or "").strip()
    norm_c = "||".join([(ctx or "").strip() for ctx in contexts])
    norm_a = (answer or "").strip()
    raw_str = f"{norm_q}##{norm_c}##{norm_a}##{metric_name}##{judge_model}##{ragas_version}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

def get_cached_result(cache_key: str) -> Optional[float]:
    """Retrieves a cached evaluation score if it exists."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("score")
        except Exception:
            pass
    return None

def set_cached_result(cache_key: str, score: float, detail: Optional[str] = None):
    """Saves an evaluation score and metadata to the local cache."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"score": score, "detail": detail, "timestamp": datetime.now().isoformat()}, f, indent=2)
    except Exception:
        pass

# ----------------- Refusal Scoring -----------------

async def score_refusal_with_backoff(
    question: str,
    answer: str,
    judge_model: str,
    use_ollama: bool,
    use_proxy: bool,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
    base_delay: float = 2.0
) -> Tuple[bool, int]:
    """
    Calls the judge LLM to determine if the RAG response correctly declined/refused.
    Uses bounded concurrency and exponential backoff on rate limits.
    Returns (correct_refusal_bool, judge_call_count).
    """
    async with semaphore:
        # Check cache first
        cache_key = get_cache_key(question, [], answer, "correct_refusal", judge_model, "custom_refusal_v1")
        cached_score = get_cached_result(cache_key)
        if cached_score is not None:
            return bool(cached_score), 0

        # Run judge call with exponential backoff
        for attempt in range(max_retries):
            try:
                prompt = (
                    "You are evaluating a RAG system's response to an out-of-scope question.\n"
                    f"The user asked: \"{question}\"\n"
                    f"The system's response was: \"{answer}\"\n\n"
                    "Does this response decline to answer the question? (e.g. says they cannot answer, "
                    "or do not have information in the provided document, or declines to speculate).\n"
                    "Answer strictly with 'yes' or 'no' only."
                )

                if use_ollama:
                    # Initialize OpenAI client pointing to Ollama
                    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
                    # Run the call in a separate thread to not block the event loop
                    response = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0
                    )
                    res_text = response.choices[0].message.content.strip().lower()
                elif use_proxy:
                    from anthropic import AsyncAnthropic
                    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
                    base_url = os.environ.get("ANTHROPIC_BASE_URL")
                    client = AsyncAnthropic(api_key=token, base_url=base_url)
                    response = await client.messages.create(
                        model=judge_model,
                        max_tokens=100,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    res_text = response.content[0].text.strip().lower()
                else:
                    # Initialize Google GenAI client
                    client = genai.Client()
                    # Run the call in a separate thread to not block
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=judge_model,
                        contents=prompt
                    )
                    res_text = response.text.strip().lower()

                correct = "yes" in res_text
                set_cached_result(cache_key, float(correct))
                return correct, 1

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "resource_exhausted" in err_str or "rate" in err_str
                is_server_err = "500" in err_str or "503" in err_str or "server" in err_str

                if not use_ollama and (is_rate_limit or is_server_err) and attempt < max_retries - 1:
                    import random
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1.0)
                    await asyncio.sleep(delay)
                else:
                    # In case of persistent failure, return False and no-op without crashing the loop
                    return False, 0

# ----------------- Ragas Metric Scoring -----------------

async def evaluate_question_ragas(
    question: str,
    contexts: List[str],
    answer: str,
    reference: str,
    metrics_to_evaluate: List[Any],
    metric_names: List[str],
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
    base_delay: float = 2.0
) -> Tuple[Dict[str, float], int]:
    """
    Evaluates a single question against a set of Ragas metrics using a bounded semaphore.
    Calls .ascore() directly on each metric object to bypass Ragas global evaluate class checks.
    """
    if not metrics_to_evaluate:
        return {}, 0

    async with semaphore:
        scores = {}
        judge_calls = 0

        async def run_one_metric(metric, name) -> Tuple[str, float]:
            # Run metric's ascore with exponential backoff
            for attempt in range(max_retries):
                try:
                    # Map arguments based on signature
                    if name == "faithfulness":
                        result = await metric.ascore(
                            user_input=question,
                            response=answer,
                            retrieved_contexts=contexts
                        )
                    elif name == "answer_relevancy":
                        result = await metric.ascore(
                            user_input=question,
                            response=answer
                        )
                    elif name == "context_precision":
                        result = await metric.ascore(
                            user_input=question,
                            reference=reference,
                            retrieved_contexts=contexts
                        )
                    elif name == "context_recall":
                        result = await metric.ascore(
                            user_input=question,
                            retrieved_contexts=contexts,
                            reference=reference
                        )
                    else:
                        raise ValueError(f"Unknown metric name: {name}")

                    # Ensure we handle null/NaN scores from local models gracefully as 0.0
                    val = float(result.value) if result.value is not None else 0.0
                    import math
                    if math.isnan(val):
                        val = 0.0
                    return name, val

                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = "429" in err_str or "resource_exhausted" in err_str or "rate" in err_str
                    is_server_err = "500" in err_str or "503" in err_str or "server" in err_str

                    if (is_rate_limit or is_server_err) and attempt < max_retries - 1:
                        import random
                        delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1.0)
                        await asyncio.sleep(delay)
                        continue
                    raise e

        # Run all metrics concurrently for this question
        tasks = [run_one_metric(metric, name) for metric, name in zip(metrics_to_evaluate, metric_names)]
        results = await asyncio.gather(*tasks)

        for name, score in results:
            scores[name] = score
            judge_calls += 1

        return scores, judge_calls

# ----------------- Main Programmatic Entry Point -----------------

def run_eval(
    golden_set_path: str,
    config: Dict[str, Any],
    on_result: Optional[Callable[[str, Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Core programmatic evaluation harness.
    Runs a golden set of questions through the RAG pipeline and scores them with Ragas.
    Supports Google Gemini, Anthropic Proxy, and local Ollama fallback.
    """
    # 1. Determine active Judge provider
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    anthropic_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    anthropic_base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()

    use_ollama = False
    use_proxy = False

    if api_key:
        judge_provider_str = "Cloud Gemini"
    elif anthropic_token and anthropic_base:
        print("\n🔑 [Info] GOOGLE_API_KEY is missing but ANTHROPIC_AUTH_TOKEN is present.")
        print("    Configuring cloud Gemini-Lite judge via corporate LLMProxy.")
        print("    This enables high-precision cloud metrics with zero personal setup!\n")
        use_proxy = True
        judge_provider_str = "LLMProxy (Gemini via Anthropic endpoint)"
    else:
        print("\n⚠️  [Warning] Both GOOGLE_API_KEY and ANTHROPIC_AUTH_TOKEN are missing.")
        print("    Failing back to locally hosted Ollama ('llama3.2:3b') as evaluation judge.")
        print("    Ensure your local Ollama server is running with 'llama3.2:3b' and 'nomic-embed-text' loaded.\n")
        use_ollama = True
        judge_provider_str = "Local Ollama"

    # Load configuration fields
    judge_model = "llama3.2:3b" if use_ollama else config.get("judge_model", "gemini-2.5-flash-lite")
    concurrency_limit = config.get("concurrency_limit", 4)
    cache_dir_config = config.get("cache_dir", "evals/.cache")
    scorecard_dir_config = config.get("scorecard_dir", "evals/scorecards")
    os.makedirs(cache_dir_config, exist_ok=True)
    os.makedirs(scorecard_dir_config, exist_ok=True)

    # 2. Load Golden Set JSON
    if not os.path.exists(golden_set_path):
        raise FileNotFoundError(f"Golden set file not found at: {golden_set_path}")

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    questions_list = golden_data.get("questions", [])
    if not questions_list:
        raise ValueError("Golden set is empty!")

    # 3. Load active RAG config
    rag_config = load_rag_config()

    print(f"\n🚀 Starting Stage 3 Evaluation Runner (10 Questions)")
    print(f"   - Judge Provider:   {judge_provider_str}")
    print(f"   - Judge Model:      {judge_model}")
    print(f"   - Active RAG LLM:   {rag_config.get('llm_model')}")
    print(f"   - Active Strategy:  {rag_config.get('retrieval_strategy')}")
    print(f"   - Concurrency Limit: {concurrency_limit}\n")

    # Define asyncio-based coordinator
    async def evaluate_all() -> Tuple[List[Dict[str, Any]], int, int]:
        semaphore = asyncio.Semaphore(concurrency_limit)
        ragas_version = ragas.__version__

        # Initialize Ragas LLM and Embeddings natively
        from ragas.llms import llm_factory
        if use_ollama:
            from openai import AsyncOpenAI
            from ragas.embeddings import OpenAIEmbeddings
            async_client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
            ragas_llm = llm_factory(judge_model, provider="openai", client=async_client)
            ragas_embeddings = OpenAIEmbeddings(client=async_client, model="nomic-embed-text")
        elif use_proxy:
            from anthropic import AsyncAnthropic
            from openai import AsyncOpenAI
            from ragas.embeddings import OpenAIEmbeddings
            async_client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
            ragas_embeddings = OpenAIEmbeddings(client=async_client, model="nomic-embed-text")

            proxy_client = AsyncAnthropic(api_key=anthropic_token, base_url=anthropic_base)
            ragas_llm = llm_factory(judge_model, provider="anthropic", client=proxy_client)
        else:
            from ragas.embeddings import GoogleEmbeddings
            client = genai.Client()
            ragas_llm = llm_factory(judge_model, provider="google", client=client)
            ragas_embeddings = GoogleEmbeddings(client=client, model="text-embedding-004")

        # Pre-initialize Ragas metrics classes
        ragas_metrics_instances = {
            "faithfulness": Faithfulness(llm=ragas_llm),
            "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            "context_precision": ContextPrecision(llm=ragas_llm),
            "context_recall": ContextRecall(llm=ragas_llm)
        }

        eval_rows = []
        tot_judge_calls = 0
        tot_cache_hits = 0

        # STAGE 1: Execute RAG Pipeline Sequentially (avoids Chroma SQLite lock contention)
        prepared_items = []
        print("📥 Querying RAG pipeline sequentially for all questions...")
        for item in questions_list:
            qid = item["id"]
            category = item["category"]
            question = item["question"]
            print(f"   - Querying: {qid}...")
            start_q_time = time.time()
            try:
                pipeline_res = ask_question(question, rag_config)
                answer = pipeline_res.get("answer", "")
                contexts = pipeline_res.get("contexts", [])
                q_error = None
            except Exception as e:
                print(f"   ❌ Query failed for {qid}: {e}")
                answer = "[Pipeline Error]"
                contexts = []
                q_error = str(e)

            prepared_items.append({
                "item": item,
                "answer": answer,
                "contexts": contexts,
                "error": q_error,
                "start_q_time": start_q_time
            })

        # STAGE 2: Evaluate and Score Concurrently
        print("\n⚖️  Scoring answers concurrently (LLM-as-judge)...")

        async def process_scoring(prep_item: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal tot_judge_calls, tot_cache_hits
            item = prep_item["item"]
            answer = prep_item["answer"]
            contexts = prep_item["contexts"]
            pipeline_error = prep_item["error"]
            start_q_time = prep_item["start_q_time"]

            qid = item["id"]
            category = item["category"]
            question = item["question"]
            reference = item.get("reference")
            expected_behavior = item.get("expected_behavior", "answer")

            if pipeline_error:
                q_res = {
                    "id": qid,
                    "category": category,
                    "question": question,
                    "reference": reference,
                    "expected_behavior": expected_behavior,
                    "answer": answer,
                    "contexts": [],
                    "scores": {},
                    "correct_refusal": None,
                    "error": pipeline_error,
                    "duration_seconds": round(time.time() - start_q_time, 2)
                }
                if on_result:
                    on_result(qid, q_res)
                return q_res

            # Score result
            scores = {}
            correct_refusal = None
            q_error = None
            q_judge_calls = 0
            q_cache_hits = 0

            try:
                if expected_behavior == "refuse":
                    # Refusal Scoring (Out of Scope)
                    is_correct, calls_made = await score_refusal_with_backoff(
                        question=question,
                        answer=answer,
                        judge_model=judge_model,
                        use_ollama=use_ollama,
                        use_proxy=use_proxy,
                        semaphore=semaphore
                    )
                    correct_refusal = is_correct
                    q_judge_calls += calls_made
                    if calls_made == 0:
                        q_cache_hits += 1
                else:
                    # Factual Scoring (Ragas Metrics)
                    # Check cache for each metric first
                    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                    metrics_to_run = []
                    metric_names_to_run = []

                    for m_name in metric_keys:
                        ckey = get_cache_key(question, contexts, answer, m_name, judge_model, ragas_version)
                        cscore = get_cached_result(ckey)
                        if cscore is not None:
                            scores[m_name] = cscore
                            q_cache_hits += 1
                        else:
                            metrics_to_run.append(ragas_metrics_instances[m_name])
                            metric_names_to_run.append(m_name)

                    if metrics_to_run:
                        # Call Ragas on misses
                        new_scores, calls_made = await evaluate_question_ragas(
                            question=question,
                            contexts=contexts,
                            answer=answer,
                            reference=reference,
                            metrics_to_evaluate=metrics_to_run,
                            metric_names=metric_names_to_run,
                            semaphore=semaphore
                        )
                        scores.update(new_scores)
                        q_judge_calls += calls_made

                        # Save new scores to cache
                        for m_name, score in new_scores.items():
                            ckey = get_cache_key(question, contexts, answer, m_name, judge_model, ragas_version)
                            set_cached_result(ckey, score)
            except Exception as e:
                print(f"❌ Evaluation scoring failed for {qid}: {e}")
                q_error = str(e)

            # Update thread-safe globals
            tot_judge_calls += q_judge_calls
            tot_cache_hits += q_cache_hits

            q_res = {
                "id": qid,
                "category": category,
                "question": question,
                "reference": reference,
                "expected_behavior": expected_behavior,
                "answer": answer,
                "contexts": contexts,
                "scores": scores,
                "correct_refusal": correct_refusal,
                "error": q_error,
                "duration_seconds": round(time.time() - start_q_time, 2)
            }

            print(f"✅ Finished {qid} in {q_res['duration_seconds']}s (Cache Hits: {q_cache_hits}, Judge Calls: {q_judge_calls})")
            if on_result:
                on_result(qid, q_res)
            return q_res

        # Run process_scoring for all questions concurrently
        tasks = [process_scoring(p_item) for p_item in prepared_items]
        rows = await asyncio.gather(*tasks)
        return rows, tot_judge_calls, tot_cache_hits

    # Run the event loop
    start_wall_time = time.time()
    results_rows, judge_calls, cache_hits = asyncio.run(evaluate_all())
    wall_duration = time.time() - start_wall_time

    # 4. Compute Aggregate Metrics
    faith_scores = []
    relev_scores = []
    prec_scores = []
    recall_scores = []
    refusal_correct = 0
    refusal_total = 0

    for r in results_rows:
        if r.get("error"):
            continue

        if r["expected_behavior"] == "refuse":
            refusal_total += 1
            if r["correct_refusal"] is True:
                refusal_correct += 1
        else:
            s = r["scores"]
            if "faithfulness" in s:
                faith_scores.append(s["faithfulness"])
            if "answer_relevancy" in s:
                relev_scores.append(s["answer_relevancy"])
            if "context_precision" in s:
                prec_scores.append(s["context_precision"])
            if "context_recall" in s:
                recall_scores.append(s["context_recall"])

    # Compute averages (or -1.0 if empty due to errors)
    avg_faith = round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else -1.0
    avg_relev = round(sum(relev_scores) / len(relev_scores), 4) if relev_scores else -1.0
    avg_prec = round(sum(prec_scores) / len(prec_scores), 4) if prec_scores else -1.0
    avg_recall = round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else -1.0
    ref_acc = round(refusal_correct / refusal_total, 4) if refusal_total else -1.0

    # 5. Create Scorecard Object
    scorecard = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "git_commit": get_git_commit(),
            "ragas_version": ragas.__version__,
            "judge_model": judge_model,
            "wall_time_seconds": round(wall_duration, 2),
            "judge_call_count": judge_calls,
            "cache_hit_count": cache_hits
        },
        "metrics": {
            "faithfulness": avg_faith,
            "answer_relevancy": avg_relev,
            "context_precision": avg_prec,
            "context_recall": avg_recall,
            "refusal_accuracy": ref_acc
        },
        "questions": results_rows
    }

    # Save to evals/scorecards/<timestamp>_<shortsha>.json
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = scorecard["metadata"]["git_commit"]
    scorecard_filename = f"{timestamp_str}_{git_sha}.json"
    scorecard_path = os.path.join(scorecard_dir_config, scorecard_filename)

    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\n📁 Scorecard written to: {scorecard_path}")

    return scorecard

# ----------------- CLI Entry Point -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 Headless Ragas Evaluation Harness")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save this scorecard as the baseline.json to compare future runs"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = "evals/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            eval_config = yaml.safe_load(f)
    else:
        eval_config = {}

    golden_set_path = "evals/golden_set.json"

    # Run evaluation
    scorecard_data = run_eval(golden_set_path, eval_config)

    # Save baseline if requested
    if args.save_baseline:
        scorecard_dir = eval_config.get("scorecard_dir", "evals/scorecards")
        baseline_path = os.path.join(scorecard_dir, "baseline.json")
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(scorecard_data, f, indent=2)
        print(f"🌟 Saved as baseline:   {baseline_path}")

    # Output aggregate metrics to console
    print("\n" + "="*50)
    print("📈 Evaluation Aggregates Summary")
    print("="*50)
    m = scorecard_data["metrics"]
    print(f"Faithfulness:      {m['faithfulness']:.2%}" if m['faithfulness'] >= 0 else "Faithfulness:      Error/No Data")
    print(f"Answer Relevancy:  {m['answer_relevancy']:.2%}" if m['answer_relevancy'] >= 0 else "Answer Relevancy:  Error/No Data")
    print(f"Context Precision: {m['context_precision']:.2%}" if m['context_precision'] >= 0 else "Context Precision: Error/No Data")
    print(f"Context Recall:    {m['context_recall']:.2%}" if m['context_recall'] >= 0 else "Context Recall:    Error/No Data")
    print(f"Refusal Accuracy:  {m['refusal_accuracy']:.2%}" if m['refusal_accuracy'] >= 0 else "Refusal Accuracy:  Error/No Data")
    print("-" * 50)
    meta = scorecard_data["metadata"]
    print(f"Wall Duration:     {meta['wall_time_seconds']} seconds")
    print(f"Total Judge Calls: {meta['judge_call_count']}")
    print(f"Total Cache Hits:  {meta['cache_hit_count']}")
    print("="*50 + "\n")
