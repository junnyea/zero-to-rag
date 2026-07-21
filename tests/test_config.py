import os
import tempfile
import pytest
import yaml
from rag.config import load_config, Settings, ConfigValidationError

def test_default_config_is_valid():
    """Verify that the default config.yaml is parsed and validated successfully."""
    # Since find_config_file looks in current dir (which is root in tests),
    # this should find and load the actual config.yaml cleanly.
    config = load_config()
    assert config.chunk_size == 800
    assert config.chunk_overlap == 150
    assert config.top_k == 3
    assert config.embed_model == "nomic-embed-text"
    assert config.llm_model == "llama3.2:3b"
    assert config.persist_dir == "./chroma_db"

def test_valid_settings_instantiation():
    """Verify that Settings can be instantiated directly with valid dictionary."""
    valid_data = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 5,
        "embed_model": "test-embed",
        "llm_model": "test-llm",
        "persist_dir": "./test_db"
    }
    settings = Settings(valid_data)
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.top_k == 5
    assert settings.embed_model == "test-embed"
    assert settings.llm_model == "test-llm"
    assert settings.persist_dir == "./test_db"

    # Test dictionary-like access
    assert settings["chunk_size"] == 1000
    assert settings.get("nonexistent", "default_val") == "default_val"
    assert settings.to_dict() == valid_data

def test_chunk_size_validation():
    """Verify chunk_size validation rules."""
    base_data = {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "top_k": 3,
        "embed_model": "embed",
        "llm_model": "llm",
        "persist_dir": "dir"
    }

    # Too small
    data_small = base_data.copy()
    data_small["chunk_size"] = 199
    with pytest.raises(ConfigValidationError, match="chunk_size must be between 200 and 2000"):
        Settings(data_small)

    # Too large
    data_large = base_data.copy()
    data_large["chunk_size"] = 2001
    with pytest.raises(ConfigValidationError, match="chunk_size must be between 200 and 2000"):
        Settings(data_large)

    # Valid boundary limits
    data_boundary_min = base_data.copy()
    data_boundary_min["chunk_size"] = 200
    data_boundary_min["chunk_overlap"] = 50  # Must be < 100
    assert Settings(data_boundary_min).chunk_size == 200

    data_boundary_max = base_data.copy()
    data_boundary_max["chunk_size"] = 2000
    assert Settings(data_boundary_max).chunk_size == 2000

def test_top_k_validation():
    """Verify top_k validation rules."""
    base_data = {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "top_k": 3,
        "embed_model": "embed",
        "llm_model": "llm",
        "persist_dir": "dir"
    }

    # Too small
    data_small = base_data.copy()
    data_small["top_k"] = 0
    with pytest.raises(ConfigValidationError, match="top_k must be between 1 and 10"):
        Settings(data_small)

    # Too large
    data_large = base_data.copy()
    data_large["top_k"] = 11
    with pytest.raises(ConfigValidationError, match="top_k must be between 1 and 10"):
        Settings(data_large)

    # Valid boundaries
    data_min = base_data.copy()
    data_min["top_k"] = 1
    assert Settings(data_min).top_k == 1

    data_max = base_data.copy()
    data_max["top_k"] = 10
    assert Settings(data_max).top_k == 10

def test_chunk_overlap_validation():
    """Verify chunk_overlap must be strictly less than chunk_size / 2."""
    base_data = {
        "chunk_size": 800,
        "chunk_overlap": 400,  # Exactly half
        "top_k": 3,
        "embed_model": "embed",
        "llm_model": "llm",
        "persist_dir": "dir"
    }

    # Equal to chunk_size / 2
    with pytest.raises(ConfigValidationError, match="chunk_overlap.*must be strictly less than half of chunk_size"):
        Settings(base_data)

    # Greater than chunk_size / 2
    data_greater = base_data.copy()
    data_greater["chunk_overlap"] = 401
    with pytest.raises(ConfigValidationError, match="chunk_overlap.*must be strictly less than half of chunk_size"):
        Settings(data_greater)

    # Strictly less than half
    data_valid = base_data.copy()
    data_valid["chunk_overlap"] = 399
    assert Settings(data_valid).chunk_overlap == 399

def test_missing_keys_validation():
    """Verify that omission of required fields raises errors."""
    base_data = {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "top_k": 3,
        "embed_model": "embed",
        "llm_model": "llm",
        "persist_dir": "dir"
    }

    for key in base_data.keys():
        data_missing = base_data.copy()
        del data_missing[key]
        with pytest.raises(ConfigValidationError, match=f"Missing required configuration key: {key}"):
            Settings(data_missing)

def test_load_from_custom_path():
    """Verify load_config loads successfully from a custom file path."""
    test_config_dict = {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 4,
        "embed_model": "custom-embed",
        "llm_model": "custom-llm",
        "persist_dir": "./custom_db"
    }

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as temp_file:
        yaml.dump(test_config_dict, temp_file)
        temp_file_path = temp_file.name

    try:
        config = load_config(temp_file_path)
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.top_k == 4
        assert config.embed_model == "custom-embed"
        assert config.llm_model == "custom-llm"
        assert config.persist_dir == "./custom_db"
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
