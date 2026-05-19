"""Tests for evaluator module."""

import pytest
from unittest.mock import MagicMock
from src.evaluator import Evaluator, EvalResult
from src.test_case import EvalCase


@pytest.fixture
def mock_rag():
    """Create a mock RAG system."""
    rag = MagicMock()
    rag.query.return_value = MagicMock(
        answer="Test answer with topics.",
        sources=[{"source": "doc.pdf", "page": 1}],
        retrieved_chunks=[MagicMock(text="Context chunk 1")]
    )
    return rag


def test_evaluator_init(mock_rag):
    """Test Evaluator initializes with RAG system."""
    evaluator = Evaluator(mock_rag)
    assert evaluator.rag == mock_rag
    assert evaluator.metrics == ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def test_run_case_returns_result(mock_rag):
    """Test run_case returns EvalResult with scores."""
    case = EvalCase(
        question="What is X?",
        ground_truth="X is a thing",
        expected_topics=["X"],
        source_hint="doc.pdf"
    )
    evaluator = Evaluator(mock_rag)
    result = evaluator.run_case(case)

    assert isinstance(result, EvalResult)
    assert result.question == "What is X?"
    assert result.answer is not None
    assert result.faithfulness >= 0
    assert result.answer_relevancy >= 0


def test_run_eval_loads_cases_from_file(mock_rag, tmp_path):
    """Test run_eval loads cases from Python file."""
    # Create test case file
    case_file = tmp_path / "test_cases.py"
    case_file.write_text('''
from src.test_case import EvalCase

CASES = [
    EvalCase(
        question="Test?",
        ground_truth="Test answer",
        expected_topics=["test"],
        source_hint="test.pdf"
    )
]
''')

    evaluator = Evaluator(mock_rag)
    results = evaluator.run_eval(str(case_file))

    assert len(results) == 1
    assert results[0].question == "Test?"