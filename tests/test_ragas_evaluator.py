"""Tests for RAGASEvaluator using ragas library."""

import pytest
from unittest.mock import MagicMock, patch

from src.evaluator import RAGASEvaluator, EvalResult
from src.test_case import EvalCase


@pytest.fixture
def mock_rag():
    """Create a mock RAG system."""
    rag = MagicMock()
    rag.query.return_value = MagicMock(
        answer="Test answer with topics.",
        retrieved_chunks=[MagicMock(text="Context chunk 1")]
    )
    return rag


@pytest.fixture
def eval_llm_config():
    """Create a mock eval LLM config."""
    config = MagicMock()
    config.eval_llm.provider = "ollama"
    config.eval_llm.model = "qwen2.5:7b"
    config.eval_llm.base_url = "http://localhost:11434/v1"
    config.eval_llm.api_key = "ollama"
    return config


def test_ragas_evaluator_init(mock_rag, eval_llm_config):
    """Test RAGASEvaluator initializes with correct attributes."""
    with patch("src.evaluator.config", eval_llm_config):
        evaluator = RAGASEvaluator(mock_rag)

    assert evaluator.rag == mock_rag
    assert evaluator.metrics == ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def test_ragas_evaluator_run_case_returns_result(mock_rag):
    """Test run_case returns EvalResult with ragas scores."""
    case = EvalCase(
        question="What is X?",
        ground_truth="X is a thing",
        expected_topics=["X"],
        source_hint="doc.pdf"
    )

    # Mock the config
    mock_config = MagicMock()
    mock_config.eval_llm.provider = "ollama"
    mock_config.eval_llm.model = "qwen2.5:7b"
    mock_config.eval_llm.base_url = "http://localhost:11434/v1"
    mock_config.eval_llm.api_key = "ollama"

    with patch("src.evaluator.config", mock_config):
        evaluator = RAGASEvaluator(mock_rag)

    result = evaluator.run_case(case)

    assert isinstance(result, EvalResult)
    assert result.question == "What is X?"
    assert result.answer is not None
    assert result.ground_truth == "X is a thing"
    assert result.contexts is not None
    assert 0.0 <= result.faithfulness <= 1.0
    assert 0.0 <= result.answer_relevancy <= 1.0
    assert 0.0 <= result.context_precision <= 1.0
    assert 0.0 <= result.context_recall <= 1.0