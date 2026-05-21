"""RAGAS evaluation integration for RAG system."""

import asyncio
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

from src.test_case import EvalCase
from src.config import config

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.embeddings import OpenAIEmbeddings
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.dataset_schema import SingleTurnSample
from openai import AsyncOpenAI


@dataclass
class EvalResult:
    """Result of evaluating a single test case.

    Attributes:
        question: The test question.
        answer: Generated answer from RAG system.
        ground_truth: Reference answer.
        contexts: Retrieved context chunks.
        sources: Formatted source references.
        faithfulness: Score 0-1 (higher = more faithful).
        answer_relevancy: Score 0-1 (higher = more relevant).
        context_precision: Score 0-1 (higher = better precision).
        context_recall: Score 0-1 (higher = better recall).
    """

    question: str
    answer: str
    ground_truth: str
    contexts: List[str]
    sources: str = ""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0


class RAGASEvaluator:
    """Evaluates RAG system responses using RAGAS with LLM-based metrics.

    This evaluator uses the ragas library to compute LLM-evaluated metrics
    including faithfulness, answer relevancy, context precision, and context recall.
    """

    def __init__(self, rag_system):
        """Initialize evaluator with RAG system and config.

        Args:
            rag_system: RAGSystem instance to evaluate.
        """
        self.rag = rag_system
        self.metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        self._llm = None
        self._embeddings = None
        self._faithfulness = None
        self._answer_relevancy = None
        self._context_precision = None
        self._context_recall = None

        # Initialize OpenAI client for Ollama (OpenAI-compatible API)
        self._async_client = AsyncOpenAI(
            api_key=config.eval_llm.api_key, base_url=config.eval_llm.base_url
        )

        # Initialize LLM wrapper for RAGAS using Ollama with async client
        self._llm = llm_factory(
            config.eval_llm.model,
            provider=config.eval_llm.provider,
            client=self._async_client,
        )

        # Initialize embeddings using RAGAS OpenAIEmbeddings function with nomic-embed-text model on Ollama
        self._embeddings = OpenAIEmbeddings(
            client=self._async_client, model="nomic-embed-text"
        )

        # Initialize metrics using the collections API
        self._faithfulness = Faithfulness(llm=self._llm)
        self._answer_relevancy = AnswerRelevancy(
            llm=self._llm, embeddings=self._embeddings
        )
        self._context_precision = ContextPrecision(llm=self._llm)
        self._context_recall = ContextRecall(llm=self._llm)

    async def _score_sample(
        self,
        sample: SingleTurnSample,
        scores: dict,
    ) -> dict:
        """Run all metric scorings and return results."""
        results = await asyncio.gather(
            self._faithfulness.ascore(
                sample.user_input, sample.response, sample.retrieved_contexts
            ),
            self._answer_relevancy.ascore(sample.user_input, sample.response),
            self._context_precision.ascore(
                sample.user_input, sample.reference, sample.retrieved_contexts
            ),
            self._context_recall.ascore(
                sample.user_input, sample.retrieved_contexts, sample.reference
            ),
            return_exceptions=True,
        )

        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            value = float(result.value) if hasattr(result, "value") else float(result)
            if value == value:  # NaN check
                scores[metric_names[i]] = value

        return scores

    def run_case(self, case: EvalCase, verbose: bool = False) -> EvalResult:
        """Run RAGAS evaluation on a single test case.

        Args:
            case: EvalCase to evaluate.
            verbose: If True, print intermediate results.

        Returns:
            EvalResult with ragas LLM-evaluated scores.
        """
        # Get RAG response
        rag_result = self.rag.query(case.question)
        answer = rag_result.answer
        sources = rag_result.sources

        # Extract contexts
        contexts = []
        for chunk in rag_result.retrieved_chunks:
            text = getattr(chunk, "text", str(chunk))
            contexts.append(text)

        if verbose:
            print(f"Question: {case.question}")
            print(f"Answer: {answer}")

        scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

        if self._llm is not None:
            try:
                sample = SingleTurnSample(
                    user_input=case.question,
                    response=answer,
                    reference=case.ground_truth,
                    retrieved_contexts=contexts,
                )

                # Apply nest_asyncio to allow nested event loops
                import nest_asyncio

                nest_asyncio.apply()

                asyncio.run(self._score_sample(sample, scores))

                if verbose:
                    print(
                        f"Faithfulness: {scores['faithfulness']:.2f}, Answer Relevancy: {scores['answer_relevancy']:.2f}"
                    )
                    print(
                        f"Context Precision: {scores['context_precision']:.2f}, Context Recall: {scores['context_recall']:.2f}"
                    )
            except Exception as e:
                import warnings

                warnings.warn(f"RAGAS evaluation failed: {e}. Using fallback scores.")

        return EvalResult(
            question=case.question,
            answer=answer,
            ground_truth=case.ground_truth,
            contexts=contexts,
            sources=sources,
            faithfulness=scores["faithfulness"],
            answer_relevancy=scores["answer_relevancy"],
            context_precision=scores["context_precision"],
            context_recall=scores["context_recall"],
        )

    def run_batch(
        self, cases: List[EvalCase], verbose: bool = False
    ) -> List[EvalResult]:
        """Run RAGAS evaluation on multiple test cases.

        Args:
            cases: List of EvalCase to evaluate.
            verbose: If True, print per-case results.

        Returns:
            List of EvalResult for each case.
        """
        results = []
        for case in cases:
            result = self.run_case(case, verbose=verbose)
            results.append(result)
        return results

    def run_eval(
        self, cases: List[EvalCase], verbose: bool = False
    ) -> List[EvalResult]:
        """Run evaluation on test cases (alias for run_batch)."""
        return self.run_batch(cases, verbose)

    def print_results(self, results: List[EvalResult]):
        """Print evaluation results as a summary table."""
        print("\n" + "=" * 80)
        print("RAGAS EVALUATION RESULTS")
        print("=" * 80)

        print(f"{'Question':<40} {'Faith':<8} {'Relev':<8} {'Prec':<8} {'Recall':<8}")
        print("-" * 80)

        for r in results:
            q = r.question[:37] + "..." if len(r.question) > 40 else r.question
            print(
                f"{q:<40} {r.faithfulness:.2f}    {r.answer_relevancy:.2f}    {r.context_precision:.2f}    {r.context_recall:.2f}"
            )
            print(f"  [Answer]: {r.answer[:100]}{'...' if len(r.answer) > 100 else ''}")
            if r.sources:
                for i, source in enumerate(r.sources, 1):
                    print(
                        f"  [{i}] {source['source']}, Page {source['page']} (score: {source['score']:.3f})"
                    )

        # Summary
        if results:
            avg_f = sum(r.faithfulness for r in results) / len(results)
            avg_r = sum(r.answer_relevancy for r in results) / len(results)
            avg_p = sum(r.context_precision for r in results) / len(results)
            avg_c = sum(r.context_recall for r in results) / len(results)

            print("-" * 80)
            print(
                f"{'AVERAGE':<40} {avg_f:.2f}    {avg_r:.2f}    {avg_p:.2f}    {avg_c:.2f}"
            )
        print("=" * 80)

    def save_results(
        self, results: List[EvalResult], output_path: str = "eval_results.json"
    ):
        """Save evaluation results to a JSON file."""
        output_data = {
            "summary": {
                "total_cases": len(results),
                "avg_faithfulness": (
                    sum(r.faithfulness for r in results) / len(results)
                    if results
                    else 0
                ),
                "avg_answer_relevancy": (
                    sum(r.answer_relevancy for r in results) / len(results)
                    if results
                    else 0
                ),
                "avg_context_precision": (
                    sum(r.context_precision for r in results) / len(results)
                    if results
                    else 0
                ),
                "avg_context_recall": (
                    sum(r.context_recall for r in results) / len(results)
                    if results
                    else 0
                ),
            },
            "cases": [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "ground_truth": r.ground_truth,
                    "sources": r.sources,
                    "contexts": r.contexts,
                    "metrics": {
                        "faithfulness": r.faithfulness,
                        "answer_relevancy": r.answer_relevancy,
                        "context_precision": r.context_precision,
                        "context_recall": r.context_recall,
                    },
                }
                for r in results
            ],
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {path.resolve()}")
