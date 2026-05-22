"""Configuration settings for RAG system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""

    model_name: str = "BAAI/bge-large-en-v1.5"
    dimension: int = 1024
    batch_size: int = 16
    instruction_prefix: str = "Represent this sentence for searching: "


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""

    semantic_top_k: int = 50
    keyword_top_k: int = 50
    final_top_k: int = 3
    rrf_k: int = 60
    # Neural reranking options
    use_neural_rerank: bool = False
    rerank_model: Optional[str] = None  # None = default model
    rerank_top_k: int = 10  # Candidates to rerank


@dataclass
class GenerationConfig:
    """Generation configuration."""

    base_url: str = "http://localhost:11434"
    model: str = "gemma4:e4b"
    temperature: float = 0.3
    max_tokens: int = 512
    stream: bool = False


@dataclass
class ChunkingConfig:
    """Chunking configuration."""

    min_chunk_size: int = 500
    max_chunk_size: int = 1500
    overlap_size: int = 200
    separators: list = field(
        default_factory=lambda: ["\n## ", "\n# ", "\n\n", "\n", " "]
    )


@dataclass
class EvalLLMConfig:
    """Configuration for the LLM used in RAGAS evaluation."""

    provider: str = "openai"  # "openai" for OpenAI-compatible API (Ollama uses this)
    model: str = "gemma4:e4b"  # Model for evaluation
    base_url: str = (
        "http://localhost:11434/v1"  # Ollama OpenAI-compatible endpoint (include /v1)
    )
    api_key: str = "ollama"  # Dummy for Ollama


@dataclass
class RAGConfig:
    """Main RAG configuration."""

    embedding: EmbeddingConfig = None
    retrieval: RetrievalConfig = None
    generation: GenerationConfig = None
    chunking: ChunkingConfig = None
    eval_llm: EvalLLMConfig = field(default_factory=EvalLLMConfig)

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.eval_llm is None:
            self.eval_llm = EvalLLMConfig()


# Global config instance
config = RAGConfig()
