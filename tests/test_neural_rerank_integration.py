import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".").absolute()))

from src.rag_system import RAGSystem
from src.config import config


def test_reranking_disabled_by_default(tmp_path):
    """Test that reranking is disabled when config is False."""
    config.retrieval.use_neural_rerank = False
    config.retrieval.final_top_k = 3

    rag = RAGSystem(source_dir=tmp_path)

    # Rerank should be None (disabled)
    assert rag.retriever.rerank is None


def test_reranking_enabled_via_config(tmp_path):
    """Test that neural reranking can be enabled via config."""
    config.retrieval.use_neural_rerank = True
    config.retrieval.rerank_model = "cross-encoder/ms-marco-MiniLM-L-12v2"

    rag = RAGSystem(source_dir=tmp_path)

    # Rerank should be NeuralRerank
    from src.neural_rerank import NeuralRerank

    assert isinstance(rag.retriever.rerank, NeuralRerank)

    # Reset config
    config.retrieval.use_neural_rerank = False
