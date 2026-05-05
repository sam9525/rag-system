"""Tests for generator module."""

import pytest
from src.generator import OllamaGenerator, OllamaConnectionError


class TestOllamaGenerator:
    """Test OllamaGenerator class."""

    def test_init_sets_config(self):
        """Test initialization sets configuration."""
        generator = OllamaGenerator()
        assert generator.config.model == "gemma4"
        assert generator.config.base_url == "http://localhost:11434"

    def test_build_prompt_format(self):
        """Test prompt building with context."""
        generator = OllamaGenerator()

        chunks = [
            {"text": "First chunk", "score": 0.9, "metadata": {"source": "doc1.pdf", "page": 1}},
            {"text": "Second chunk", "score": 0.8, "metadata": {"source": "doc2.pdf", "page": 2}},
        ]

        prompt = generator._build_prompt("Test question?", chunks)

        assert "Test question?" in prompt
        assert "First chunk" in prompt
        assert "[Source: doc1.pdf, Page 1]" in prompt

    def test_generate_requires_ollama_running(self):
        """Test generate fails gracefully if Ollama not running."""
        generator = OllamaGenerator()
        chunks = [{"text": "test", "score": 0.9, "metadata": {"source": "test.pdf", "page": 1}}]
        # This will fail if Ollama not running - test error handling
        try:
            response = generator.generate("test", chunks)
            # If it succeeds, Ollama is running
        except OllamaConnectionError:
            pass  # Expected if Ollama not running