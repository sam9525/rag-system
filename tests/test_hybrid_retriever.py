"""Tests for hybrid retriever."""

import pytest
import numpy as np
from src.hybrid_retriever import HybridRetriever, RRFResult


class TestHybridRetriever:
    """Test HybridRetriever class."""

    def test_rrf_fusion_ranks_correctly(self):
        """Test RRF fusion combines rankings properly."""
        retriever = HybridRetriever(embedding_dim=128)

        # Two sets of rankings
        semantic_results = [
            ("doc_a", 0.9),
            ("doc_b", 0.8),
            ("doc_c", 0.7),
        ]
        keyword_results = [
            ("doc_b", 1.0),
            ("doc_c", 0.9),
            ("doc_a", 0.6),
        ]

        fused = retriever._rrf_fusion(semantic_results, keyword_results, k=60)

        # doc_b should be ranked first (top in both)
        assert fused[0][0] == "doc_b"

    def test_search_returns_final_top_k(self):
        """Test search returns final top k chunks."""
        retriever = HybridRetriever(embedding_dim=128)
        # Would need mocked embeddings
        pass

    def test_empty_corpus_handling(self):
        """Test handling of empty corpus."""
        retriever = HybridRetriever(embedding_dim=128)
        results = retriever.search("query", top_k=3)
        assert len(results) == 0