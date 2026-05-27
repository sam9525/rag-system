import pytest
from src.retrieval.rrf_fusion import rrf_fusion, RRFResult


class TestRRFFusion:
    def test_rrf_basic(self):
        """Test basic RRF fusion with two result sets."""
        semantic = [(0, 0.9), (1, 0.8), (2, 0.7)]
        keyword = [(0, 0.9), (1, 0.8), (2, 0.7)]

        results = rrf_fusion(semantic, keyword, k=60)

        # Chunk 0 should be first (ranked 1st in both)
        assert results[0][0] == 0
        # Chunk 1 should be second (ranked 2nd in both)
        assert results[1][0] == 1

    def test_rrf_different_ranks(self):
        """Test when items have different ranks in each list."""
        semantic = [(0, 0.9), (1, 0.8), (2, 0.7)]  # 0, 1, 2
        keyword = [(2, 0.9), (1, 0.8), (0, 0.7)]  # 2, 1, 0 (reversed)

        results = rrf_fusion(semantic, keyword, k=60)
        indices = [r[0] for r in results]

        # All three should be present
        assert set(indices) == {0, 1, 2}

    def test_rrf_single_result_set(self):
        """Test RRF with only semantic results."""
        semantic = [(0, 0.9), (1, 0.8)]
        keyword = []

        results = rrf_fusion(semantic, keyword, k=60)

        assert len(results) == 2
        assert results[0][0] == 0
        assert results[1][0] == 1

    def test_rrf_empty(self):
        """Test RRF with empty inputs."""
        results = rrf_fusion([], [], k=60)
        assert results == []

    def test_rrf_k_parameter(self):
        """Test different k values affect scoring."""
        semantic = [(0, 0.9), (1, 0.8)]
        keyword = [(0, 0.9), (1, 0.8)]

        results_k1 = rrf_fusion(semantic, keyword, k=1)
        results_k60 = rrf_fusion(semantic, keyword, k=60)

        # Same ordering but different scores
        assert [r[0] for r in results_k1] == [r[0] for r in results_k60]
        # Lower k = higher scores
        assert results_k1[0][1] > results_k60[0][1]


class TestRRFResult:
    def test_rrf_result_dataclass(self):
        result = RRFResult(
            text="test chunk",
            score=0.95,
            metadata={"source": "doc.pdf"},
            chunk_index=2,
            semantic_score=0.9,
            keyword_score=0.8,
        )
        assert result.text == "test chunk"
        assert result.score == 0.95
        assert result.chunk_index == 2
        assert result.semantic_score == 0.9
        assert result.keyword_score == 0.8

    def test_rrf_result_defaults(self):
        result = RRFResult(text="test", score=0.5, metadata={})
        assert result.chunk_index == 0
        assert result.semantic_score is None
        assert result.keyword_score is None
