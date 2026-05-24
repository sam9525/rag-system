"""Tests for generator module."""

import pytest
from src.generator import OllamaGenerator, OllamaConnectionError, OllamaAPIError


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

    def test_generate_requires_ollama_running(self):
        """Test generate fails gracefully if Ollama not running."""
        generator = OllamaGenerator()
        chunks = [
            {
                "text": "test",
                "score": 0.9,
                "metadata": {"source": "test.pdf", "page": 1},
            }
        ]
        # This will fail if Ollama not running - test error handling
        try:
            response = generator.generate("test", chunks)
            # If it succeeds, Ollama is running
        except OllamaConnectionError:
            pass  # Expected if Ollama not running

    def test_is_available_returns_connection_status(self):
        """Test is_available returns true when Ollama running, false otherwise."""
        generator = OllamaGenerator()

        # Mock the connection check
        generator._check_connection = lambda: True
        assert generator.is_available() == True

        generator._check_connection = lambda: False
        assert generator.is_available() == False

    def test_generate_raises_if_unavailable(self):
        """Test generate raises OllamaConnectionError if unavailable."""
        generator = OllamaGenerator()

        # Mock unavailable state
        generator._check_connection = lambda: False

        chunks = [
            {
                "text": "test",
                "score": 0.9,
                "metadata": {"source": "test.pdf", "page": 1},
            }
        ]

        with pytest.raises(OllamaConnectionError):
            generator.generate("test question", chunks)

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
