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
    "persist_dir": "./chroma_db"
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
    Returns None if valid, or a friendly error message string if invalid.
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

    return ""
