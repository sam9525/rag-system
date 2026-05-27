"""Ollama API client with injectable interface for testing."""

from typing import Protocol
import requests


class OllamaClient(Protocol):
    """Protocol for LLM API clients — enables mock injection in tests."""

    def check_connection(self) -> bool:
        """Check if the Ollama server is reachable."""
        ...

    def generate(
        self,
        model: str,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate a response from the LLM."""
        ...


class RequestsOllamaClient:
    """requests-based implementation of OllamaClient."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url
        self.timeout = timeout

    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate a response using the Ollama API."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")
