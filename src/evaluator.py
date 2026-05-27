"""RAGAS evaluation integration for RAG system."""

import asyncio
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

from src.test_case import EvalCase
from src.config import EvalLLMConfig

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevance,
)
from ragas.embeddings import OpenAIEmbeddings
from ragas.dataset_schema import SingleTurnSample
from openai import AsyncOpenAI


@dataclass
class EvalResult:
    """Result of evaluating a single test case.

    Attributes:
        question: The test question.
        answer: Generated answer from RAG system.
        contexts: Retrieved context chunks.
        sources: Formatted source references.
        faithfulness: Score 0-1 (higher = more faithful).
        answer_relevancy: Score 0-1 (higher = more relevant).
        context_relevance: Score 0-1 (higher = better relevance).
    """

    question: str
    answer: str
    contexts: List[str]
    sources: str = ""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_relevance: float = 0.0


class RAGASEvaluator:
    """Evaluates RAG system responses using RAGAS with LLM-based metrics.

    This evaluator uses the ragas library to compute LLM-evaluated metrics
    including faithfulness, answer relevancy, and context relevance.
    """

    def __init__(self, rag_system, rerank_mode: str = "hybrid"):
        """Initialize evaluator with RAG system and config.

        Args:
            rag_system: RAGSystem instance to evaluate.
            rerank_mode: One of "rrf", "neural", or "hybrid".
                - "rrf": Only RRF fusion, no neural reranking
                - "neural": Only semantic + neural reranking, skip keyword search
                - "hybrid": Full pipeline (default)
        """
        self.rag = rag_system
        self.rerank_mode = rerank_mode
        self.metrics = ["faithfulness", "answer_relevancy", "context_relevance"]
        self._llm = None
        self._embeddings = None
        self._faithfulness = None
        self._answer_relevancy = None
        self._context_relevance = None

        # Initialize config
        self._eval_config = EvalLLMConfig()

        # Initialize OpenAI client for Ollama (OpenAI-compatible API)
        self._async_client = AsyncOpenAI(
            api_key=self._eval_config.api_key, base_url=self._eval_config.base_url
        )

        # Initialize LLM wrapper for RAGAS using Ollama with async client
        self._llm = llm_factory(
            self._eval_config.model,
            provider=self._eval_config.provider,
            client=self._async_client,
        )

        # Initialize embeddings using RAGAS OpenAIEmbeddings function with nomic-embed-text model on Ollama
        self._embeddings = OpenAIEmbeddings(
            client=self._async_client, model="nomic-embed-text"
        )

        # Initialize metrics
        self._faithfulness = Faithfulness(llm=self._llm)
        self._answer_relevancy = AnswerRelevancy(
            llm=self._llm, embeddings=self._embeddings
        )
        self._context_relevance = ContextRelevance(llm=self._llm)

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
            self._context_relevance.ascore(
                sample.user_input, sample.retrieved_contexts
            ),
            return_exceptions=True,
        )

        metric_names = ["faithfulness", "answer_relevancy", "context_relevance"]
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
        rag_result = self.rag.query(case.question, rerank_mode=self.rerank_mode)
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
            "context_relevance": 0.0,
        }

        if self._llm is not None:
            try:
                sample = SingleTurnSample(
                    user_input=case.question,
                    response=answer,
                    retrieved_contexts=contexts,
                )

                # Apply nest_asyncio to allow nested event loops
                import nest_asyncio

                nest_asyncio.apply()

                asyncio.run(self._score_sample(sample, scores))

                if verbose:
                    print(
                        f"Faithfulness: {scores['faithfulness']:.2f}, "
                        f"Answer Relevancy: {scores['answer_relevancy']:.2f}, "
                        f"Context Relevance: {scores['context_relevance']:.2f}"
                    )
            except Exception as e:
                import warnings

                warnings.warn(f"RAGAS evaluation failed: {e}. Using fallback scores.")

        return EvalResult(
            question=case.question,
            answer=answer,
            contexts=contexts,
            sources=sources,
            faithfulness=scores["faithfulness"],
            answer_relevancy=scores["answer_relevancy"],
            context_relevance=scores["context_relevance"],
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
        print(f"[Evaluator] Starting batch evaluation of {len(cases)} cases...")
        results = []
        for i, case in enumerate(cases, 1):
            print(f"[Evaluator] [{i}/{len(cases)}] Processing case...")
            result = self.run_case(case, verbose=verbose)
            results.append(result)
            print(f"[Evaluator] [{i}/{len(cases)}] Case complete")
        print(f"[Evaluator] Batch evaluation complete: {len(results)} results")
        return results

    def run_eval(
        self, cases: List[EvalCase], verbose: bool = False
    ) -> List[EvalResult]:
        """Run evaluation on test cases (alias for run_batch)."""
        return self.run_batch(cases, verbose)

    def query_baseline(self, question: str) -> str:
        """Generate answer WITHOUT retrieval - pure model knowledge.

        This simulates asking the LLM without any RAG context,
        allowing comparison between grounded vs ungrounded answers.

        Args:
            question: User's question

        Returns:
            Generated response with NO context chunks provided
        """
        try:
            # Pass empty chunks - model should either say it lacks info
            # or generate from training knowledge (depending on system prompt behavior)
            return self.rag.generator.generate(question, chunks=[], rag=False)
        except Exception as e:
            return f"Error generating baseline: {e}"

    def run_baseline_case(self, case: EvalCase, verbose: bool = False) -> EvalResult:
        """Run baseline evaluation on a single test case (no retrieval).

        Args:
            case: EvalCase to evaluate.
            verbose: If True, print intermediate results.

        Returns:
            EvalResult with baseline scores (no retrieval).
        """
        # Get baseline response (no retrieval)
        answer = self.query_baseline(case.question)

        # Empty contexts for baseline
        contexts = []

        if verbose:
            print(f"[Baseline] Question: {case.question}")
            print(f"[Baseline] Answer: {answer[:100]}...")

        scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_relevance": 0.0,
        }

        if self._llm is not None:
            try:
                sample = SingleTurnSample(
                    user_input=case.question,
                    response=answer,
                    retrieved_contexts=contexts,
                )

                import nest_asyncio

                nest_asyncio.apply()

                asyncio.run(self._score_sample(sample, scores))

                if verbose:
                    print(
                        f"[Baseline] Faithfulness: {scores['faithfulness']:.2f}, "
                        f"Answer Relevancy: {scores['answer_relevancy']:.2f}, "
                        f"Context Relevance: {scores['context_relevance']:.2f}"
                    )
            except Exception as e:
                import warnings

                warnings.warn(f"RAGAS evaluation failed for baseline: {e}")

        return EvalResult(
            question=case.question,
            answer=answer,
            contexts=contexts,
            sources=[],
            faithfulness=scores["faithfulness"],
            answer_relevancy=scores["answer_relevancy"],
            context_relevance=scores["context_relevance"],
        )

    def run_baseline_batch(
        self, cases: List[EvalCase], verbose: bool = False
    ) -> List[EvalResult]:
        """Run baseline evaluation on multiple test cases (no retrieval).

        Args:
            cases: List of EvalCase to evaluate.
            verbose: If True, print per-case results.

        Returns:
            List of EvalResult for each baseline case.
        """
        print(
            f"[Baseline] Starting baseline evaluation of {len(cases)} cases (no retrieval)..."
        )
        results = []
        for i, case in enumerate(cases, 1):
            print(f"[Baseline] [{i}/{len(cases)}] Processing case...")
            result = self.run_baseline_case(case, verbose=verbose)
            results.append(result)
            print(f"[Baseline] [{i}/{len(cases)}] Case complete")
        print(f"[Baseline] Batch evaluation complete: {len(results)} results")
        return results

    def print_results(self, results: List[EvalResult]):
        """Print evaluation results as a summary table."""
        print("\n" + "=" * 80)
        print("RAGAS EVALUATION RESULTS")
        print("=" * 80)

        print(f"{'Question':<50} {'Faith':<8} {'Relev':<8} {'CtxRel':<8}")
        print("-" * 80)

        for r in results:
            q = r.question[:47] + "..." if len(r.question) > 50 else r.question
            print(
                f"{q:<50} {r.faithfulness:.2f}    {r.answer_relevancy:.2f}    {r.context_relevance:.2f}"
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
            avg_cr = sum(r.context_relevance for r in results) / len(results)

            print("-" * 80)
            print(f"{'AVERAGE':<50} {avg_f:.2f}    {avg_r:.2f}    {avg_cr:.2f}")
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
                "avg_context_relevance": (
                    sum(r.context_relevance for r in results) / len(results)
                    if results
                    else 0
                ),
            },
            "cases": [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "sources": r.sources,
                    "contexts": r.contexts,
                    "metrics": {
                        "faithfulness": r.faithfulness,
                        "answer_relevancy": r.answer_relevancy,
                        "context_relevance": r.context_relevance,
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
