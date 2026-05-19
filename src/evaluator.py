"""RAGAS evaluation integration for RAG system."""

from dataclasses import dataclass
from typing import List
import importlib.util

from src.test_case import EvalCase


@dataclass
class EvalResult:
    """Result of evaluating a single test case.

    Attributes:
        question: The test question.
        answer: Generated answer from RAG system.
        ground_truth: Reference answer.
        contexts: Retrieved context chunks.
        faithfulness: Score 0-1 (higher = more faithful).
        answer_relevancy: Score 0-1 (higher = more relevant).
        context_precision: Score 0-1 (higher = better precision).
        context_recall: Score 0-1 (higher = better recall).
    """

    question: str
    answer: str
    ground_truth: str
    contexts: List[str]
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0


class Evaluator:
    """Evaluates RAG system responses using RAGAS metrics."""

    def __init__(self, rag_system):
        """Initialize evaluator with RAG system.

        Args:
            rag_system: RAGSystem instance to evaluate.
        """
        self.rag = rag_system
        self.metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def run_case(self, case: EvalCase, verbose: bool = False) -> EvalResult:
        """Run evaluation on a single test case.

        Args:
            case: EvalCase to evaluate.
            verbose: If True, print intermediate results.

        Returns:
            EvalResult with scores.
        """
        # Get RAG response
        result = self.rag.query(case.question)
        answer = result.answer

        # Extract contexts
        contexts = []
        for chunk in result.retrieved_chunks:
            text = getattr(chunk, "text", str(chunk))
            contexts.append(text)

        # Calculate RAGAS metrics
        # For now, use simple heuristic-based scoring
        # Full RAGAS integration requires OpenAI API and ragas package
        faithfulness_score = self._compute_faithfulness(answer, contexts)
        relevancy_score = self._compute_answer_relevancy(answer, case.question, case.expected_topics)
        precision_score = self._compute_context_precision(contexts, case.expected_topics)
        recall_score = self._compute_context_recall(contexts, case.ground_truth)

        if verbose:
            print(f"Question: {case.question}")
            print(f"Answer: {answer}")
            print(f"Faithfulness: {faithfulness_score:.2f}, Relevancy: {relevancy_score:.2f}")

        return EvalResult(
            question=case.question,
            answer=answer,
            ground_truth=case.ground_truth,
            contexts=contexts,
            faithfulness=faithfulness_score,
            answer_relevancy=relevancy_score,
            context_precision=precision_score,
            context_recall=recall_score
        )

    def _compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Compute faithfulness score based on answer-context alignment.

        Args:
            answer: Generated answer.
            contexts: Retrieved context chunks.

        Returns:
            Score 0-1.
        """
        if not contexts or not answer:
            return 0.0

        # Simple heuristic: check overlap between answer and context
        answer_lower = answer.lower()
        context_text = " ".join(contexts).lower()

        # Count answer words that appear in context
        answer_words = [w for w in answer_lower.split() if len(w) > 3]
        if not answer_words:
            return 1.0

        matches = sum(1 for w in answer_words if w in context_text)
        return min(1.0, matches / len(answer_words))

    def _compute_answer_relevancy(self, answer: str, question: str, expected_topics: List[str]) -> float:
        """Compute answer relevancy score.

        Args:
            answer: Generated answer.
            question: Original question.
            expected_topics: Topics that should appear.

        Returns:
            Score 0-1.
        """
        if not answer:
            return 0.0

        answer_lower = answer.lower()

        # Check if expected topics appear in answer
        topic_matches = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
        topic_score = topic_matches / len(expected_topics) if expected_topics else 1.0

        # Check if answer is non-trivial (not just repeating question)
        answer_words = set(answer_lower.split())
        question_words = set(question.lower().split())
        novel_words = len(answer_words - question_words) / len(answer_words) if answer_words else 0

        # Weighted combination
        return 0.6 * topic_score + 0.4 * min(1.0, novel_words * 2)

    def _compute_context_precision(self, contexts: List[str], expected_topics: List[str]) -> float:
        """Compute context precision score.

        Args:
            contexts: Retrieved context chunks.
            expected_topics: Topics that should be present.

        Returns:
            Score 0-1.
        """
        if not contexts:
            return 0.0

        # Check how many expected topics appear in contexts
        context_text = " ".join(contexts).lower()
        matches = sum(1 for topic in expected_topics if topic.lower() in context_text)
        return matches / len(expected_topics) if expected_topics else 1.0

    def _compute_context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """Compute context recall score based on ground truth.

        Args:
            contexts: Retrieved context chunks.
            ground_truth: Reference answer.

        Returns:
            Score 0-1.
        """
        if not contexts:
            return 0.0

        context_text = " ".join(contexts).lower()
        gt_words = [w for w in ground_truth.lower().split() if len(w) > 3]

        if not gt_words:
            return 1.0

        matches = sum(1 for w in gt_words if w in context_text)
        return min(1.0, matches / len(gt_words))

    def run_eval(self, cases_path: str) -> List[EvalResult]:
        """Run evaluation on all test cases in a file.

        Args:
            cases_path: Path to Python file with CASES list.

        Returns:
            List of EvalResult for each case.
        """
        # Load test cases from file
        spec = importlib.util.spec_from_file_location("cases", cases_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cases = module.CASES

        results = []
        for case in cases:
            result = self.run_case(case)
            results.append(result)

        return results

    def print_results(self, results: List[EvalResult]):
        """Print evaluation results as a summary table.

        Args:
            results: List of EvalResult from run_eval.
        """
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        # Header
        print(f"{'Question':<40} {'Faith':<8} {'Relev':<8} {'Prec':<8} {'Recall':<8}")
        print("-" * 80)

        for r in results:
            q = r.question[:37] + "..." if len(r.question) > 40 else r.question
            print(f"{q:<40} {r.faithfulness:.2f}    {r.answer_relevancy:.2f}    {r.context_precision:.2f}    {r.context_recall:.2f}")

        # Summary
        avg_f = sum(r.faithfulness for r in results) / len(results)
        avg_r = sum(r.answer_relevancy for r in results) / len(results)
        avg_p = sum(r.context_precision for r in results) / len(results)
        avg_c = sum(r.context_recall for r in results) / len(results)

        print("-" * 80)
        print(f"{'AVERAGE':<40} {avg_f:.2f}    {avg_r:.2f}    {avg_p:.2f}    {avg_c:.2f}")
        print("=" * 80)

    def generate_candidates(self, top_k: int = 10) -> List[EvalCase]:
        """Generate candidate test cases from indexed chunks.

        This uses the existing RAG system to generate questions from chunks,
        then formats them as candidates for human review.

        Args:
            top_k: Number of candidate cases to generate.

        Returns:
            List of candidate EvalCase objects.
        """
        # Placeholder - uses existing RAG query to generate questions
        # Will be implemented if needed
        return []