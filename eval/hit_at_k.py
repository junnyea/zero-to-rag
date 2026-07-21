import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple

# Add parent directory to sys.path so we can import rag
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rag.config import load_config, save_config, validate_config
from rag.ingest import ingest_file, reset_index, get_index_stats, load_file_content
from rag.query import retrieve_context_chunks

GOLD_SET_PATH = os.path.join(os.path.dirname(__file__), "gold.jsonl")
SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_docs")

def load_gold_set(path: str = GOLD_SET_PATH) -> List[Dict[str, str]]:
    """Loads Q&A pairs from gold.jsonl."""
    gold_set = []
    if not os.path.exists(path):
        print(f"Error: Gold set file not found at {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_set.append(json.loads(line.strip()))
    return gold_set

def evaluate_retrieval(k: int, config: Dict[str, Any], gold_set: List[Dict[str, str]]) -> Tuple[float, float]:
    """
    Evaluates retrieval for the gold set under the given config.
    Returns:
        - hit_rate: fraction of questions where expected source was in the top-k chunks.
        - mean_length: average character length of all retrieved chunks.
    """
    hits = 0
    total_chunks_len = 0
    total_chunks_count = 0

    # Ensure config's top_k is set to our target k
    eval_config = config.copy()
    eval_config["top_k"] = k

    for idx, item in enumerate(gold_set):
        question = item["question"]
        expected = item["expected_source"]

        chunks, status = retrieve_context_chunks(question, eval_config)
        if status != "ok":
            print(f"  [Eval Error] Retrieval failed for question {idx+1}: {status}")
            continue

        # Check for hit
        retrieved_sources = [chunk["source"] for chunk in chunks]
        is_hit = expected in retrieved_sources
        if is_hit:
            hits += 1

        # Calculate character lengths
        for chunk in chunks:
            total_chunks_len += len(chunk["content"])
            total_chunks_count += 1

    hit_rate = hits / len(gold_set) if gold_set else 0.0
    mean_length = total_chunks_len / total_chunks_count if total_chunks_count else 0.0

    return hit_rate, mean_length

def reindex_corpus(config: Dict[str, Any]) -> None:
    """Wipes the database and re-indexes all files in sample_docs/."""
    persist_dir = config["persist_dir"]
    reset_index(persist_dir)

    if not os.path.exists(SAMPLE_DOCS_DIR):
        print(f"Error: Sample docs directory {SAMPLE_DOCS_DIR} does not exist.")
        return

    for filename in os.listdir(SAMPLE_DOCS_DIR):
        fpath = os.path.join(SAMPLE_DOCS_DIR, filename)
        if os.path.isfile(fpath):
            try:
                content = load_file_content(fpath, filename)
                ingest_file(content, filename, config)
            except Exception as e:
                print(f"  [Reindex Error] Failed to ingest {filename}: {e}")

def run_parameter_experiments(k: int) -> None:
    """
    Runs evaluation over a parameter grid:
      chunk_size ∈ {300, 800, 1500}
      chunk_overlap ∈ {0, 150}
    Records metrics, prints a table, and promotes the winner to config.yaml.
    """
    gold_set = load_gold_set()
    if not gold_set:
        print("No gold Q&A pairs available to evaluate.")
        return

    # Original config to restore or update
    original_config = load_config()

    # Grid search parameters
    sizes = [300, 800, 1500]
    overlaps = [0, 150]

    experiments = []

    print("\n" + "="*70)
    print(f"🚀 Running Grid Search Experiments (Evaluating hit@{k})")
    print("="*70)

    for size in sizes:
        for overlap in overlaps:
            # Adjust overlap if it violates physical constraints (overlap < size / 2)
            active_overlap = overlap
            if overlap >= size / 2:
                active_overlap = size // 3 # Safe fallback (e.g. 100 for size 300)
                print(f"⚠️  Adjusted overlap {overlap} -> {active_overlap} for size {size} to pass validation.")

            exp_config = original_config.copy()
            exp_config["chunk_size"] = size
            exp_config["chunk_overlap"] = active_overlap

            # Validate before running
            validation_err = validate_config(exp_config)
            if validation_err:
                print(f"❌ Skipping experiment Size={size}, Overlap={active_overlap}: {validation_err}")
                continue

            print(f"🧪 Testing Configuration: Chunk Size = {size}, Overlap = {active_overlap}...")
            # Re-index
            reindex_corpus(exp_config)
            # Evaluate
            hit_rate, mean_len = evaluate_retrieval(k, exp_config, gold_set)

            stats = get_index_stats(exp_config["persist_dir"])
            total_chunks = stats["total_chunks"]

            experiments.append({
                "chunk_size": size,
                "chunk_overlap": active_overlap,
                "total_chunks": total_chunks,
                "hit_rate": hit_rate,
                "mean_length": mean_len
            })
            print(f"   👉 Results: hit@{k} = {hit_rate:.1%}, Mean Chunk Len = {mean_len:.1f} chars, Total Chunks = {total_chunks}")

    # Print Tabular Summary
    print("\n" + "="*70)
    print("📊 Experiment Summary Table")
    print("="*70)
    print(f"{'Chunk Size':<12} | {'Overlap':<10} | {'Total Chunks':<12} | {'hit@' + str(k):<10} | {'Mean Chunk Len':<15}")
    print("-" * 70)
    for exp in experiments:
        print(
            f"{exp['chunk_size']:<12} | {exp['chunk_overlap']:<10} | {exp['total_chunks']:<12} | "
            f"{exp['hit_rate']:<10.1%} | {exp['mean_length']:<15.1f}"
        )
    print("="*70)

    if not experiments:
        print("No valid experiments completed.")
        return

    # Find the winner (Maximize hit_rate, then minimize mean_length as tie-breaker)
    winner = max(experiments, key=lambda x: (x["hit_rate"], -x["mean_length"]))

    print("\n🏆 Winning Parameter Configuration:")
    print(f"   - Chunk Size: {winner['chunk_size']}")
    print(f"   - Chunk Overlap: {winner['chunk_overlap']}")
    print(f"   - hit@{k}: {winner['hit_rate']:.1%}")
    print(f"   - Mean Chunk Length: {winner['mean_length']:.1f} characters")

    # Update config.yaml with winner
    updated_config = original_config.copy()
    updated_config["chunk_size"] = winner["chunk_size"]
    updated_config["chunk_overlap"] = winner["chunk_overlap"]
    save_config(updated_config)
    print(f"📝 Winner promoted and written to config.yaml as the new defaults!")

    # Reindex one final time with winning params
    print("\n🔄 Re-indexing index with final winning parameters...")
    reindex_corpus(updated_config)
    print("✅ Completed. App is ready with winning default parameters.")

def main():
    parser = argparse.ArgumentParser(description="Stage 1 RAG Evaluation Harness (hit@k)")
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Retrieval depth to evaluate (default 3)"
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Trigger grid search over chunking parameters"
    )
    args = parser.parse_args()

    gold_set = load_gold_set()
    if not gold_set:
        sys.exit(1)

    if args.run_experiments:
        run_parameter_experiments(args.k)
    else:
        config = load_config()
        print(f"Evaluating active configuration: Chunk Size = {config['chunk_size']}, Overlap = {config['chunk_overlap']}, k = {args.k}")
        hit_rate, mean_len = evaluate_retrieval(args.k, config, gold_set)
        stats = get_index_stats(config["persist_dir"])
        print("\n" + "="*50)
        print("📈 Evaluation Results")
        print("="*50)
        print(f"Active Index Total Chunks: {stats['total_chunks']}")
        print(f"Evaluation hit@{args.k} Score:   {hit_rate:.1%}")
        print(f"Mean Retrieved Chunk Length: {mean_len:.1f} characters")
        print("="*50)

if __name__ == "__main__":
    main()
