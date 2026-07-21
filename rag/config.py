import os
import yaml
from typing import Dict, Any, Union

class ConfigValidationError(ValueError):
    """Exception raised when configuration parameters are invalid."""
    pass

class Settings:
    """Settings object to represent parsed and validated configurations."""
    def __init__(self, config_dict: Dict[str, Any]):
        self.chunk_size = config_dict.get("chunk_size")
        self.chunk_overlap = config_dict.get("chunk_overlap")
        self.top_k = config_dict.get("top_k")
        self.embed_model = config_dict.get("embed_model")
        self.llm_model = config_dict.get("llm_model")
        self.persist_dir = config_dict.get("persist_dir")
        self.validate()

    def validate(self) -> None:
        required_keys = ["chunk_size", "chunk_overlap", "top_k", "embed_model", "llm_model", "persist_dir"]
        for key in required_keys:
            if getattr(self, key, None) is None:
                raise ConfigValidationError(f"Missing required configuration key: {key}")

        # Check types and try to convert to integers if they are not
        if not isinstance(self.chunk_size, int):
            try:
                self.chunk_size = int(self.chunk_size)
            except (ValueError, TypeError):
                raise ConfigValidationError(f"chunk_size must be an integer, got {type(self.chunk_size).__name__}")

        if not isinstance(self.chunk_overlap, int):
            try:
                self.chunk_overlap = int(self.chunk_overlap)
            except (ValueError, TypeError):
                raise ConfigValidationError(f"chunk_overlap must be an integer, got {type(self.chunk_overlap).__name__}")

        if not isinstance(self.top_k, int):
            try:
                self.top_k = int(self.top_k)
            except (ValueError, TypeError):
                raise ConfigValidationError(f"top_k must be an integer, got {type(self.top_k).__name__}")

        # Range validations
        if not (200 <= self.chunk_size <= 2000):
            raise ConfigValidationError(f"chunk_size must be between 200 and 2000 (inclusive), got {self.chunk_size}")

        if not (1 <= self.top_k <= 10):
            raise ConfigValidationError(f"top_k must be between 1 and 10 (inclusive), got {self.top_k}")

        if self.chunk_overlap >= self.chunk_size / 2:
            raise ConfigValidationError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than half of chunk_size ({self.chunk_size} / 2 = {self.chunk_size / 2})"
            )

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to a dictionary."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "embed_model": self.embed_model,
            "llm_model": self.llm_model,
            "persist_dir": self.persist_dir,
        }

def find_config_file() -> str:
    """Find the config.yaml file by walking up from the current directory."""
    # Start looking in the current working directory
    cwd_path = os.path.join(os.getcwd(), "config.yaml")
    if os.path.exists(cwd_path):
        return cwd_path

    # Try looking relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    rel_path = os.path.join(parent_dir, "config.yaml")
    if os.path.exists(rel_path):
        return rel_path

    raise FileNotFoundError("Could not find config.yaml in working directory or project root.")

def load_config(config_path: str = None) -> Settings:
    """Load config.yaml, parse and validate the settings.

    Args:
        config_path: Path to the config yaml file. If None, resolves automatically.

    Returns:
        A validated Settings object.
    """
    if config_path is None:
        config_path = find_config_file()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read or parse configuration file: {e}")

    if not isinstance(config_dict, dict):
        raise ConfigValidationError("Configuration file must contain a top-level YAML dictionary.")

    return Settings(config_dict)
