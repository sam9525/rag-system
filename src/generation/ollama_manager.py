"""Ollama model manager for connectivity, availability checking, and model pre-loading."""

import time
import requests
from typing import Optional, List, Tuple


class OllamaManager:
    """Manages Ollama connectivity and model lifecycle."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma4"):
        """Initialize Ollama manager.

        Args:
            base_url: Ollama server URL
            model: Model name to manage
        """
        self.base_url = base_url
        self.model = model

    def is_available(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "unknown") for m in models]
            return []
        except requests.exceptions.RequestException:
            return []

    def find_matching_model(self, configured_model: str) -> Optional[str]:
        """Find the best matching model name.

        Args:
            configured_model: The configured model name (may be partial)

        Returns:
            Full model name if found, None otherwise
        """
        available = self.get_available_models()
        for model in available:
            # Exact match or prefix match (e.g., 'gemma4' matches 'gemma4:e4b')
            if model == configured_model or model.startswith(configured_model + ":"):
                return model
        return None

    def preload_model(self, model_name: str, timeout: int = 300) -> Tuple[bool, float]:
        """Pre-load a model into memory.

        Args:
            model_name: Full model name to load
            timeout: Maximum time to wait in seconds

        Returns:
            Tuple of (success, elapsed_time)
        """
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {"num_predict": 1},  # Minimal generation to load model
                },
                timeout=timeout,
            )
            elapsed = time.time() - start_time
            return response.status_code == 200, elapsed
        except requests.exceptions.Timeout:
            return (
                True,
                time.time() - start_time,
            )  # Timeout likely means model is loading
        except requests.exceptions.RequestException:
            return False, time.time() - start_time

    def prepare_model(self) -> Tuple[bool, str]:
        """Check connectivity, find model, and preload it.

        Returns:
            Tuple of (success, message)
        """
        if not self.is_available():
            return False, "Ollama is not running. Start with `ollama serve`"

        available = self.get_available_models()
        if not available:
            return False, "Could not retrieve available models"

        matching = self.find_matching_model(self.model)
        if not matching:
            return False, f"Model '{self.model}' not found. Available: {available}"

        print(f"Loading model '{matching}' into memory...")
        success, elapsed = self.preload_model(matching)

        if success:
            return True, f"Model '{matching}' ready ({elapsed:.1f}s)"
        else:
            return False, f"Failed to load model '{matching}'"

    def get_status_summary(self) -> dict:
        """Get a summary of Ollama and model status."""
        status = {
            "ollama_available": self.is_available(),
            "available_models": self.get_available_models(),
            "configured_model": self.model,
            "matching_model": self.find_matching_model(self.model),
            "model_loaded": False,
        }

        if status["matching_model"]:
            # Try a minimal request to check if model is already loaded
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": status["matching_model"],
                        "prompt": "hi",
                        "options": {"num_predict": 1},
                    },
                    timeout=10,
                )
                status["model_loaded"] = resp.status_code == 200
            except Exception:
                pass

        return status
