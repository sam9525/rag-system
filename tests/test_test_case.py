"""Tests for EvalCase dataclass."""

from src.evaluation.test_case import EvalCase


def test_eval_case_creation():
    """Test that EvalCase can be created with required fields."""
    case = EvalCase(question="What is prompt injection?")
    assert case.question == "What is prompt injection?"


def test_eval_case_repr():
    """Test string representation for debugging."""
    case = EvalCase(question="Test question?")
    assert "Test question?" in repr(case)
