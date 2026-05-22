import pytest
from src.neural_rerank import NeuralRerank, NoOpRerank, RerankResult


class TestNeuralRerank:
    def test_rerank_initialization(self):
        rerank = NeuralRerank()
        assert rerank.model_name == "cross-encoder/ms-marco-MiniLM-L-12v2"
        assert rerank.default_model == "cross-encoder/ms-marco-MiniLM-L-12v2"

    def test_rerank_returns_top_k(self):
        rerank = NoOpRerank()  # Skip model download in unit test
        query = "What is prompt injection?"
        chunks = ["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"]
        results = rerank(query, chunks, top_k=2)
        assert len(results) == 2
        assert all(hasattr(r, "text") for r in results)
        assert all(hasattr(r, "rerank_score") for r in results)

    def test_rerank_with_insufficient_chunks(self):
        rerank = NoOpRerank()
        query = "test query"
        chunks = ["Only one chunk"]
        results = rerank(query, chunks, top_k=5)
        assert len(results) == 1  # Returns what's available


class TestNoOpRerank:
    def test_noop_returns_original_order(self):
        rerank = NoOpRerank()
        query = "test"
        chunks = ["a", "b", "c"]
        results = rerank(query, chunks, top_k=3)
        assert [r.text for r in results] == ["a", "b", "c"]

    def test_noop_decreasing_scores(self):
        rerank = NoOpRerank()
        query = "test"
        chunks = ["a", "b", "c"]
        results = rerank(query, chunks, top_k=3)
        # First should have highest score (1.0), last should have lowest
        assert results[0].rerank_score > results[1].rerank_score
        assert results[1].rerank_score > results[2].rerank_score

    def test_noop_preserves_original_index(self):
        rerank = NoOpRerank()
        query = "test"
        chunks = ["a", "b", "c"]
        results = rerank(query, chunks, top_k=3)
        for i, result in enumerate(results):
            assert result.original_index == i


class TestRerankResult:
    def test_rerank_result_dataclass(self):
        result = RerankResult(text="test", rerank_score=0.95, original_index=2)
        assert result.text == "test"
        assert result.rerank_score == 0.95
        assert result.original_index == 2
