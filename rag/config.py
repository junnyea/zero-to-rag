import os
import yaml
from typing import Dict, Any

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

DEFAULT_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 150,
    "top_k": 3,
    "embed_model": "nomic-embed-text",
    "llm_model": "llama3.2:3b",
    "persist_dir": "./chroma_db",
    # Stage 2 Additions
    "retrieval_strategy": "adaptive",
    "reranker": "cohere",
    "rerank_model": "rerank-v3.5",
    "candidate_k": 20,
    "rewrite_trigger_score": 0.5,
    "max_rewrites": 2,
    "multi_query_n": 3,
    "query_llm_model": "llama3.2:3b",
    "trace_dir": "./traces",
    "trace_keep": 200
}

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Loads config.yaml from disk, falling back to defaults if not found."""
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if not config:
                return DEFAULT_CONFIG.copy()
            # Merge defaults for any missing keys
            for k, v in DEFAULT_CONFIG.items():
                if k not in config:
                    config[k] = v
            return config
    except Exception:
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any], config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Saves configuration dict back to config.yaml on disk."""
    try:
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)
    except Exception as e:
        raise IOError(f"Failed to save config to {config_path}: {e}")

def validate_config(config: Dict[str, Any]) -> str:
    """
    Validates configuration values.
    Returns empty string if valid, or a friendly error message string if invalid.
    """
    try:
        chunk_size = int(config.get("chunk_size", 800))
        chunk_overlap = int(config.get("chunk_overlap", 150))
        top_k = int(config.get("top_k", 3))
    except (ValueError, TypeError):
        return "Configuration parameters 'chunk_size', 'chunk_overlap', and 'top_k' must be integers."

    if not (200 <= chunk_size <= 2000):
        return f"Chunk size ({chunk_size}) must be between 200 and 2000 characters."

    if not (0 <= chunk_overlap <= 400):
        return f"Chunk overlap ({chunk_overlap}) must be between 0 and 400 characters."

    if chunk_overlap >= chunk_size / 2:
        return f"Chunk overlap ({chunk_overlap}) must be less than half of the chunk size ({chunk_size / 2:.0f})."

    if not (1 <= top_k <= 10):
        return f"Top K ({top_k}) must be between 1 and 10."

    if not config.get("embed_model"):
        return "Embedding model must be specified."

    if not config.get("llm_model"):
        return "LLM model must be specified."

    # Stage 2 validations
    strategy = config.get("retrieval_strategy", "adaptive")
    if strategy not in ("adaptive", "plain", "multi_query", "hyde"):
        return f"Invalid retrieval strategy '{strategy}'. Must be one of: 'adaptive', 'plain', 'multi_query', 'hyde'."

    reranker = config.get("reranker", "cohere")
    if reranker not in ("cohere", "none", "local"):
        return f"Invalid reranker '{reranker}'. Must be one of: 'cohere', 'none', 'local'."

    try:
        candidate_k = int(config.get("candidate_k", 20))
        if not (5 <= candidate_k <= 50):
            return f"Candidate pool size ({candidate_k}) must be between 5 and 50."
    except (ValueError, TypeError):
        return "Candidate pool size 'candidate_k' must be an integer."

    try:
        trigger_score = float(config.get("rewrite_trigger_score", 0.5))
        if not (0.0 <= trigger_score <= 1.0):
            return f"Rewrite trigger score ({trigger_score}) must be between 0.0 and 1.0."
    except (ValueError, TypeError):
        return "Rewrite trigger score 'rewrite_trigger_score' must be a float."

    try:
        max_rewrites = int(config.get("max_rewrites", 2))
        if not (0 <= max_rewrites <= 3):
            return f"Max rewrites ({max_rewrites}) must be between 0 and 3."
    except (ValueError, TypeError):
        return "Max rewrites 'max_rewrites' must be an integer."

    try:
        multi_query_n = int(config.get("multi_query_n", 3))
        if not (2 <= multi_query_n <= 5):
            return f"Multi-query count ({multi_query_n}) must be between 2 and 5."
    except (ValueError, TypeError):
        return "Multi-query count 'multi_query_n' must be an integer."

    try:
        trace_keep = int(config.get("trace_keep", 200))
        if not (50 <= trace_keep <= 1000):
            return f"Trace retention count ({trace_keep}) must be between 50 and 1000."
    except (ValueError, TypeError):
        return "Trace retention count 'trace_keep' must be an integer."

    return ""
