"""Tests for EvalCase dataclass."""

from src.test_case import EvalCase


def test_eval_case_creation():
    """Test that EvalCase can be created with required fields."""
    case = EvalCase(
        question="What is prompt injection?",
        ground_truth="Prompt injection is a vulnerability where attackers inject malicious instructions",
        expected_topics=["prompt", "injection", "vulnerability"],
        source_hint="OWASP LLM Top 10"
    )
    assert case.question == "What is prompt injection?"
    assert case.ground_truth.startswith("Prompt injection")
    assert len(case.expected_topics) == 3
    assert case.source_hint == "OWASP LLM Top 10"


def test_eval_case_repr():
    """Test string representation for debugging."""
    case = EvalCase(
        question="Test question?",
        ground_truth="Test answer",
        expected_topics=["topic1"],
        source_hint="Doc1"
    )
    assert "Test question?" in repr(case)
    assert "Test answer" in repr(case)