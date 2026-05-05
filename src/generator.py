"""Ollama-based generator using gemma4."""

from typing import List, Dict
import requests

from src.config import config as global_config


class OllamaConnectionError(Exception):
    """Raised when Ollama is not available."""
    pass


class OllamaGenerationError(Exception):
    """Raised when generation fails."""
    pass


class OllamaGenerator:
    """Generates responses using Ollama with gemma4."""

    SYSTEM_PROMPT = """You are a precise RAG assistant. Answer ONLY using the provided context chunks.

CONSTRAINTS:
1. Answer ONLY from the provided context. Do not use external knowledge.
2. If information is not in the context, say "I don't have this information from the provided documents."
3. Always cite sources using [Source: filename, page N] after statements.
4. Keep answers concise: 2-4 sentences for simple questions, up to 8 sentences for complex ones.
5. Do not invent, infer, or hallucinate information not explicitly in the context.
6. If the context is insufficient to fully answer, acknowledge what you can and cannot determine.
7. Maintain professional, neutral tone.
8. Format: plain text with markdown for clarity. Use bullet points only if > 3 items.

RESPONSE FORMAT:
- Direct answer first
- [Source citations in brackets]
- Brief follow-up context if needed"""

    def __init__(self, model: str = None, base_url: str = None, config=None):
        """Initialize generator with configuration."""
        self.config = config if config is not None else global_config.generation

        if model:
            self.config.model = model
        if base_url:
            self.config.base_url = base_url

    def _check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _build_prompt(self, question: str, chunks: List[Dict]) -> str:
        """Build the full prompt with context chunks."""

        chunks_text = ""
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", "unknown")
            page = chunk.get("metadata", {}).get("page", "?")
            score = chunk.get("score", 0)

            chunks_text += f"""
---
Chunk {i} (Score: {score:.3f}):
{chunk['text']}
[Source: {source}, Page {page}]
---
"""

        prompt = f"""CONTEXT CHUNKS:
{chunks_text}

USER QUERY: {question}

INSTRUCTIONS:
- Answer using ONLY the provided context chunks
- Cite sources after each statement
- If info is missing, say so clearly
- Be concise and precise

ANSWER:"""

        return prompt

    def generate(self, question: str, chunks: List[Dict]) -> str:
        """Generate response using Ollama.

        Args:
            question: User's question
            chunks: Retrieved context chunks

        Returns:
            Generated response string
        """
        if not self._check_connection():
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Please ensure Ollama is running (ollama serve)."
            )

        prompt = self._build_prompt(question, chunks)

        try:
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "system": self.SYSTEM_PROMPT,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": self.config.stream
                },
                timeout=120
            )

            if response.status_code != 200:
                raise OllamaGenerationError(f"Generation failed: {response.status_code}")

            result = response.json()
            return result.get("response", "")

        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/show",
                params={"name": self.config.model},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}