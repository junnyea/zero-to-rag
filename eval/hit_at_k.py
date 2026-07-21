import os
import sys
import json
import argparse
from typing import List, Dict, Any

# Adjust path to import from rag module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import load_config
from rag.ingest import ingest_file, reset_index, get_chroma_collection_and_embeddings

def load_gold_set(gold_path: str = "eval/gold.jsonl") -> List[Dict[str, Any]]:
    """Load the gold evaluation set from a JSONL file."""
    gold_set = []
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold dataset not found at {gold_path}")

    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gold_set.append(json.loads(line))
    return gold_set

def evaluate_hit_at_k(k: int = 3, gold_path: str = "eval/gold.jsonl") -> Dict[str, Any]:
    """Calculate the hit@k metric on the gold set.

    Model-free, cheap, and isolates retrieval quality from generation quality.
    """
    gold_set = load_gold_set(gold_path)
    hits = 0
    total = len(gold_set)
    total_retrieved_length = 0
    retrieved_chunks_count = 0

    vector_store, _ = get_chroma_collection_and_embeddings()

    for item in gold_set:
        question = item["question"]
        expected_source = item["expected_source"]

        # Prefix search query for nomic-embed-text
        prefixed_query = f"search_query: {question}"

        # Search
        results = vector_store.similarity_search_with_score(prefixed_query, k=k)

        # Extract source filenames from top-k retrieved chunks
        retrieved_sources = [doc.metadata.get("source", "unknown") for doc, _ in results]

        # Check for a hit
        if expected_source in retrieved_sources:
            hits += 1

        # Track retrieved chunk lengths for character span average reporting
        for doc, _ in results:
            raw_content = doc.page_content
            if raw_content.startswith("search_document: "):
                raw_content = raw_content[len("search_document: "):]
            total_retrieved_length += len(raw_content)
            retrieved_chunks_count += 1

    hit_rate = float(hits) / total if total > 0 else 0.0
    mean_chunk_len = float(total_retrieved_length) / retrieved_chunks_count if retrieved_chunks_count > 0 else 0.0

    return {
        "hit_rate": hit_rate,
        "hits": hits,
        "total": total,
        "mean_chunk_len": round(mean_chunk_len, 2)
    }

def run_parameter_sweep(sample_dir: str = "sample_docs", gold_path: str = "eval/gold.jsonl"):
    """Run a parameter sweep over chunk size and overlap, reporting hit@3."""
    print("==================================================================")
    print("🚀 STARTING CHUNKING PARAMETER SWEEP EXPERIMENT (hit@3)")
    print("==================================================================")

    # Clean corpus list
    files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith((".txt", ".md", ".pdf"))]

    # Grid search: chunk_size x chunk_overlap
    sizes = [300, 800, 1500]
    overlaps = [0, 150]

    sweep_rows = []

    for size in sizes:
        for overlap in overlaps:
            # Overlap must be strictly less than size / 2
            if overlap >= size / 2:
                continue

            print(f"Testing size={size}, overlap={overlap}...")

            # 1. Temporarily write config on disk
            with open("config.yaml", "w", encoding="utf-8") as f:
                f.write(f"""# Sweep config
chunk_size: {size}
chunk_overlap: {overlap}
top_k: 3
embed_model: "nomic-embed-text"
llm_model: "llama3.2:3b"
persist_dir: "./chroma_db"
""")

            # 2. Reset collection
            reset_index()

            # 3. Ingest files
            for file_path in files:
                ingest_file(file_path)

            # 4. Evaluate
            results = evaluate_hit_at_k(k=3, gold_path=gold_path)

            sweep_rows.append({
                "chunk_size": size,
                "chunk_overlap": overlap,
                "hit_rate": results["hit_rate"],
                "hits_pct": f"{results['hits']}/{results['total']}",
                "mean_len": results["mean_chunk_len"]
            })

    # Restore default configuration (chunk_size: 800, overlap: 150)
    print("Restoring default configuration (size=800, overlap=150)...")
    with open("config.yaml", "w", encoding="utf-8") as f:
        f.write("""# Default configurations for Local Doc Q&A RAG Pipeline (Phase 1)
chunk_size: 800
chunk_overlap: 150
top_k: 3
embed_model: "nomic-embed-text"
llm_model: "llama3.2:3b"
persist_dir: "./chroma_db"
""")

    # Re-index with defaults
    reset_index()
    for file_path in files:
        ingest_file(file_path)

    # Render Markdown table
    md_content = f"""# Parameter Sweep Results: Chunk Size vs. Overlap

Evaluated on the ACME Corp gold dataset (`{gold_path}`) consisting of {len(sweep_rows[0]['hits_pct'].split('/')[1]) if sweep_rows else 8} Q&A cases.

| Chunk Size | Chunk Overlap | Hit@3 Rate | Hits | Mean Retrieved Chunk Length (chars) |
|---|---|---|---|---|
"""
    for r in sweep_rows:
        md_content += f"| {r['chunk_size']} | {r['chunk_overlap']} | **{r['hit_rate']:.2f}** | {r['hits_pct']} | {r['mean_len']} |\n"

    # Write file
    os.makedirs("eval", exist_ok=True)
    with open("eval/sweep_results.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + md_content)
    print("==================================================================")
    print("🎉 Parameter sweep complete! Saved to 'eval/sweep_results.md'.")
    print("==================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hit@k or run a chunking sweep.")
    parser.add_argument("--k", type=int, default=3, help="Retrieval depth to evaluate (default: 3)")
    parser.add_argument("--sweep", action="store_true", help="Run the full parameter sweep experiment")
    args = parser.parse_args()

    if args.sweep:
        run_parameter_sweep()
    else:
        results = evaluate_hit_at_k(k=args.k)
        print(f"=========================================================")
        print(f"📊 RAG RETRIEVAL EVALUATION RESULTS (hit@{args.k})")
        print(f"=========================================================")
        print(f"- Hit@{args.k} Rate: {results['hit_rate']:.4f} ({results['hits']}/{results['total']})")
        print(f"- Mean Retrieved Chunk Length: {results['mean_chunk_len']} characters")
        print(f"=========================================================")
