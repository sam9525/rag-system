"""Test case dataclass for RAG evaluation."""

from dataclasses import dataclass
from typing import List


@dataclass
class EvalCase:
    """A test case for evaluating RAG system responses.

    Attributes:
        question: The question to ask the RAG system.
        ground_truth: Reference answer that represents ideal response.
        expected_topics: Key concepts that should appear in a good answer.
        source_hint: Which document/section this relates to (for debugging).
    """

    question: str
    ground_truth: str
    expected_topics: List[str]
    source_hint: str = ""