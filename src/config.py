"""Configuration settings for RAG system."""

from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


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


@dataclass
class GenerationConfig:
    """Generation configuration."""
    base_url: str = "http://localhost:11434"
    model: str = "gemma4"
    temperature: float = 0.3
    max_tokens: int = 512
    stream: bool = False


@dataclass
class ChunkingConfig:
    """Chunking configuration."""
    min_chunk_size: int = 200
    max_chunk_size: int = 1500
    overlap_size: int = 200


@dataclass
class RAGConfig:
    """Main RAG configuration."""
    embedding: EmbeddingConfig = None
    retrieval: RetrievalConfig = None
    generation: GenerationConfig = None
    chunking: ChunkingConfig = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()


# Global config instance
config = RAGConfig()