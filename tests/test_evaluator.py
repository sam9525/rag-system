"""Tests for evaluator module using RAGAS."""

import pytest
from unittest.mock import MagicMock
from src.evaluator import RAGASEvaluator, EvalResult
from src.test_case import EvalCase


@pytest.fixture
def mock_rag():
    """Create a mock RAG system."""
    rag = MagicMock()
    rag.query.return_value = MagicMock(
        answer="Test answer with topics.",
        sources=[{"source": "doc.pdf", "page": 1, "score": 0.95}],
        retrieved_chunks=[MagicMock(text="Context chunk 1")]
    )
    return rag


def test_evaluator_init(mock_rag):
    """Test RAGASEvaluator initializes with RAG system."""
    evaluator = RAGASEvaluator(mock_rag)
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
    evaluator = RAGASEvaluator(mock_rag)
    result = evaluator.run_case(case)

    assert isinstance(result, EvalResult)
    assert result.question == "What is X?"
    assert result.answer is not None
    assert result.faithfulness >= 0
    assert result.answer_relevancy >= 0


def test_run_batch_processes_cases(mock_rag):
    """Test run_batch evaluates multiple cases."""
    cases = [
        EvalCase(
            question="Test 1?",
            ground_truth="Answer 1",
            expected_topics=["test1"],
            source_hint="test.pdf"
        ),
        EvalCase(
            question="Test 2?",
            ground_truth="Answer 2",
            expected_topics=["test2"],
            source_hint="test.pdf"
        ),
    ]

    evaluator = RAGASEvaluator(mock_rag)
    results = evaluator.run_batch(cases)

    assert len(results) == 2
    assert results[0].question == "Test 1?"
    assert results[1].question == "Test 2?"


def test_run_eval_is_alias_for_run_batch(mock_rag):
    """Test run_eval is an alias for run_batch."""
    cases = [
        EvalCase(
            question="Test?",
            ground_truth="Answer",
            expected_topics=["test"],
            source_hint="test.pdf"
        )
    ]

    evaluator = RAGASEvaluator(mock_rag)
    results = evaluator.run_eval(cases)

    assert len(results) == 1
    assert results[0].question == "Test?"


def test_print_results_formats_table(mock_rag):
    """Test print_results outputs formatted table."""
    cases = [
        EvalCase(
            question="Test question?",
            ground_truth="Test answer",
            expected_topics=["test"],
            source_hint="test.pdf"
        )
    ]

    evaluator = RAGASEvaluator(mock_rag)
    results = evaluator.run_case(cases[0])

    # Should not raise
    evaluator.print_results([results])