import pytest
import requests
from unittest.mock import patch, MagicMock
from rag.preflight import check_preflight
from rag.config import load_config

@patch("requests.get")
def test_preflight_ollama_offline(mock_get):
    """Verify preflight behavior when the Ollama daemon is not running."""
    # Mock requests.get to throw connection error
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

    status = check_preflight()

    assert status["ollama_online"] is False
    assert status["embed_model_available"] is False
    assert status["llm_model_available"] is False
    assert any("not running" in err for err in status["errors"])
    assert any("ollama serve" in inst for inst in status["instructions"])

@patch("requests.get")
def test_preflight_all_models_present(mock_get):
    """Verify preflight passes when Ollama is online and all models are available."""
    config = load_config()
    embed_target = config.embed_model
    llm_target = config.llm_model

    # Mock the two requests: ping check and api/tags check
    mock_ping_response = MagicMock()
    mock_ping_response.status_code = 200

    mock_tags_response = MagicMock()
    mock_tags_response.status_code = 200
    mock_tags_response.json.return_value = {
        "models": [
            {"name": f"{embed_target}:latest"},
            {"name": f"{llm_target}:latest"}
        ]
    }

    mock_get.side_effect = [mock_ping_response, mock_tags_response]

    status = check_preflight()

    assert status["ollama_online"] is True
    assert status["embed_model_available"] is True
    assert status["llm_model_available"] is True
    assert len(status["errors"]) == 0
    assert len(status["instructions"]) == 0

@patch("requests.get")
def test_preflight_models_missing(mock_get):
    """Verify preflight status and recovery instructions when models are missing."""
    # Mock the two requests: ping check and api/tags check
    mock_ping_response = MagicMock()
    mock_ping_response.status_code = 200

    mock_tags_response = MagicMock()
    mock_tags_response.status_code = 200
    mock_tags_response.json.return_value = {
        "models": [
            {"name": "some-other-model:latest"}
        ]
    }

    mock_get.side_effect = [mock_ping_response, mock_tags_response]

    status = check_preflight()

    assert status["ollama_online"] is True
    assert status["embed_model_available"] is False
    assert status["llm_model_available"] is False
    assert len(status["errors"]) == 2
    assert any("ollama pull nomic-embed-text" in inst for inst in status["instructions"])
    assert any("ollama pull llama3.2:3b" in inst for inst in status["instructions"])

@patch("requests.get")
def test_preflight_flexible_model_matching(mock_get):
    """Verify preflight successfully matches model names with loose tag/suffix naming."""
    mock_ping_response = MagicMock()
    mock_ping_response.status_code = 200

    mock_tags_response = MagicMock()
    mock_tags_response.status_code = 200
    # Available model names match targets loosely (e.g. without tags or different tags)
    mock_tags_response.json.return_value = {
        "models": [
            {"name": "nomic-embed-text"}, # configured: nomic-embed-text
            {"name": "llama3.2:latest"}   # configured: llama3.2:3b, bases are both llama3.2
        ]
    }

    mock_get.side_effect = [mock_ping_response, mock_tags_response]

    status = check_preflight()

    assert status["ollama_online"] is True
    assert status["embed_model_available"] is True
    assert status["llm_model_available"] is True
    assert len(status["errors"]) == 0
