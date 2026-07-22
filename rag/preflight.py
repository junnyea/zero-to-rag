import urllib.request
import json
from typing import Dict, Any, List

OLLAMA_URL = "http://localhost:11434"

def check_ollama_status() -> Dict[str, Any]:
    """
    Checks if Ollama is running and if the configured models are pulled.
    Returns a dictionary of statuses:
    {
        "server_running": bool,
        "models_status": {
            "embed_model": { "status": bool, "message": str, "command": str },
            "llm_model": { "status": bool, "message": str, "command": str }
        },
        "available_models": list of str
    }
    """
    status = {
        "server_running": False,
        "models_status": {
            "embed_model": {"status": False, "message": "Ollama server is not running.", "command": "ollama serve"},
            "llm_model": {"status": False, "message": "Ollama server is not running.", "command": "ollama serve"}
        },
        "available_models": []
    }

    # 1. Check if Ollama server is running
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                status["server_running"] = True
                data = json.loads(response.read().decode())
                models_data = data.get("models", [])
                # Extract all names
                available_models = [m.get("name") for m in models_data if m.get("name")]
                # Also extract bare names (without :latest, etc.)
                all_model_names = []
                for m in available_models:
                    all_model_names.append(m)
                    if ":" in m:
                        all_model_names.append(m.split(":")[0])
                status["available_models"] = available_models
    except Exception as e:
        # Server is down
        return status

    # 2. Check the specific models from configuration
    # Read current config to know what models to check
    from rag.config import load_config
    config = load_config()
    embed_model_target = config.get("embed_model", "nomic-embed-text")
    llm_model_target = config.get("llm_model", "llama3.2:3b")

    # Match embeddings model
    embed_matched = False
    for available_name in status["available_models"]:
        # Match "nomic-embed-text" with "nomic-embed-text:latest" or exact
        if available_name == embed_model_target or available_name.startswith(f"{embed_model_target}:") or embed_model_target.startswith(f"{available_name}:"):
            embed_matched = True
            break
        # Also handle specific latest mapping
        if embed_model_target == "nomic-embed-text" and available_name == "nomic-embed-text:latest":
            embed_matched = True
            break

    if embed_matched:
        status["models_status"]["embed_model"] = {
            "status": True,
            "message": f"Embedding model '{embed_model_target}' is ready.",
            "command": ""
        }
    else:
        status["models_status"]["embed_model"] = {
            "status": False,
            "message": f"Embedding model '{embed_model_target}' is missing.",
            "command": f"ollama pull {embed_model_target}"
        }

    # Match LLM model
    llm_matched = False
    for available_name in status["available_models"]:
        if available_name == llm_model_target or available_name.startswith(f"{llm_model_target}:") or llm_model_target.startswith(f"{available_name}:"):
            llm_matched = True
            break
        if llm_model_target == "llama3.2:3b" and available_name == "llama3.2:3b:latest":
            llm_matched = True
            break

    if llm_matched:
        status["models_status"]["llm_model"] = {
            "status": True,
            "message": f"LLM model '{llm_model_target}' is ready.",
            "command": ""
        }
    else:
        status["models_status"]["llm_model"] = {
            "status": False,
            "message": f"LLM model '{llm_model_target}' is missing.",
            "command": f"ollama pull {llm_model_target}"
        }

    return status

def check_cohere_key_status() -> Dict[str, Any]:
    """
    Checks if COHERE_API_KEY is present in environment variables.
    """
    import os
    key = os.environ.get("COHERE_API_KEY", "").strip()
    is_present = len(key) > 0
    return {
        "key_present": is_present,
        "message": "Cohere API key is loaded." if is_present else "COHERE_API_KEY environment variable is not set."
    }
