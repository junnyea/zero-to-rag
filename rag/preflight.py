import requests
from typing import Dict, Any, List
from rag.config import load_config

def check_preflight() -> Dict[str, Any]:
    """Perform system preflight checks on Ollama and required models.

    Returns:
        A dictionary containing the status of Ollama connection, embedding model,
        and LLM model, along with any error messages and actionable recovery instructions.
    """
    config = load_config()
    embed_model_target = config.embed_model
    llm_model_target = config.llm_model
    ollama_url = "http://localhost:11434"

    status = {
        "ollama_online": False,
        "embed_model_available": False,
        "llm_model_available": False,
        "errors": [],
        "instructions": []
    }

    # 1. Check if Ollama server is running
    try:
        response = requests.get(ollama_url, timeout=3.0)
        if response.status_code == 200:
            status["ollama_online"] = True
        else:
            status["errors"].append(f"Ollama server returned unexpected status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        status["errors"].append("Ollama server is not running.")
        status["instructions"].append("Please start the Ollama daemon by running 'ollama serve' in a separate terminal.")
        return status

    # 2. Check if models are pulled
    try:
        tags_response = requests.get(f"{ollama_url}/api/tags", timeout=3.0)
        if tags_response.status_code != 200:
            status["errors"].append(f"Failed to query Ollama models endpoint. Status code: {tags_response.status_code}")
            return status

        models_data = tags_response.json()
        available_models = [m["name"] for m in models_data.get("models", [])]

        # Helper function to match model names flexibly (handling tags like :latest, :3b, etc.)
        def model_matches(target: str, available: List[str]) -> bool:
            # Direct match
            if target in available:
                return True

            # Normalize target and available names (remove :latest or tags for loose matching)
            target_base = target.split(":")[0] if ":" in target else target

            for model_name in available:
                model_base = model_name.split(":")[0] if ":" in model_name else model_name
                # If bases match, we can check tag loose equivalents
                if target_base == model_base:
                    # Treat direct base match or latest match as compatible
                    return True
            return False

        # Verify embedding model availability
        if model_matches(embed_model_target, available_models):
            status["embed_model_available"] = True
        else:
            status["errors"].append(f"Embedding model '{embed_model_target}' is missing.")
            status["instructions"].append(f"Please pull the embedding model by running: 'ollama pull {embed_model_target}'")

        # Verify LLM model availability
        if model_matches(llm_model_target, available_models):
            status["llm_model_available"] = True
        else:
            status["errors"].append(f"LLM model '{llm_model_target}' is missing.")
            status["instructions"].append(f"Please pull the LLM model by running: 'ollama pull {llm_model_target}'")

    except requests.exceptions.RequestException as e:
        status["errors"].append(f"Error querying Ollama API: {str(e)}")
    except ValueError:
        status["errors"].append("Failed to parse JSON response from Ollama API.")

    return status
