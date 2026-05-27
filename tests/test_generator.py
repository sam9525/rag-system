"""Tests for generator module."""

import pytest
from src.generation.generator import (
    OllamaGenerator,
    OllamaConnectionError,
    OllamaAPIError,
)
from src.generation.ollama_client import OllamaClient


class MockOllamaClient:
    """Mock client for testing without network."""

    def __init__(self, available: bool = True, response: str = "Mock response"):
        self._available = available
        self._response = response
        self.check_connection_called = False

    def check_connection(self) -> bool:
        self.check_connection_called = True
        return self._available

    def generate(
        self,
        model: str,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        return self._response


class TestOllamaGenerator:
    """Test OllamaGenerator class."""

    def test_init_sets_config(self):
        """Test initialization sets configuration."""
        generator = OllamaGenerator()
        assert generator.config.model == "gemma4:e4b"
        assert generator.config.base_url == "http://localhost:11434"

    def test_build_prompt_format(self):
        """Test prompt building with numbered context."""
        generator = OllamaGenerator()

        chunks = [
            {
                "text": "First chunk",
                "score": 0.9,
                "metadata": {"source": "doc1.pdf", "page": 1},
            },
            {
                "text": "Second chunk",
                "score": 0.8,
                "metadata": {"source": "doc2.pdf", "page": 2},
            },
        ]

        prompt = generator._build_prompt("Test question?", chunks)

        assert "Test question?" in prompt
        assert "First chunk" in prompt
        # Should use numbered format [1], [2] not verbose format
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "Chunk 1" in prompt

    def test_is_available_returns_connection_status(self):
        """Test is_available returns true when Ollama running, false otherwise."""
        mock = MockOllamaClient(available=True)
        generator = OllamaGenerator(client=mock)

        assert generator.is_available() is True
        assert mock.check_connection_called is True

        mock2 = MockOllamaClient(available=False)
        generator2 = OllamaGenerator(client=mock2)

        assert generator2.is_available() is False

    def test_connection_error_on_refused(self):
        """Test that refused connection raises OllamaConnectionError."""
        generator = OllamaGenerator(base_url="http://localhost:19999")

        chunks = [
            {"text": "test", "score": 0.9, "metadata": {"source": "t.pdf", "page": 1}}
        ]

        with pytest.raises(OllamaConnectionError):
            generator.generate("test", chunks)

    def test_citations_are_numbered(self):
        """Test that citations use numbered format [1], [2] not verbose format."""
        generator = OllamaGenerator()

        chunks = [
            {
                "text": "First chunk content",
                "score": 0.9,
                "metadata": {"source": "doc1.pdf", "page": 5},
            },
            {
                "text": "Second chunk content",
                "score": 0.8,
                "metadata": {"source": "doc2.pdf", "page": 10},
            },
        ]

        prompt = generator._build_prompt("Test question?", chunks)

        # Should NOT contain verbose citation format
        assert "[Source:" not in prompt
        assert "Page 5" not in prompt

        # Should contain numbered citations
        assert "[1]" in prompt
        assert "[2]" in prompt

        # Chunks should be labeled with numbers
        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt

    def test_generator_uses_injected_client(self):
        """Test generator uses the injected client instead of making live requests."""
        from src.system.config import GenerationConfig

        mock = MockOllamaClient(available=True, response="test output")
        config = GenerationConfig()
        generator = OllamaGenerator(config=config, client=mock)

        chunks = [
            {
                "text": "context",
                "score": 0.9,
                "metadata": {"source": "x.pdf", "page": 1},
            }
        ]
        result = generator.generate("question", chunks)

        assert result == "test output"
        assert mock.check_connection_called is True

    def test_generate_raises_when_client_unavailable(self):
        """Test generate raises OllamaConnectionError when client reports unavailable."""
        from src.system.config import GenerationConfig

        mock = MockOllamaClient(available=False)
        config = GenerationConfig()
        generator = OllamaGenerator(config=config, client=mock)

        with pytest.raises(OllamaConnectionError):
            generator.generate("question", [])
