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

    def test_inject_mock_embedding_manager(self):
        """Test HybridRetriever accepts injected embedding manager."""
        class MockEmbeddingManager:
            dimension = 128
            model_name = "mock"
            def embed_text(self, text):
                import numpy as np
                return np.zeros(128)
            def embed_batch(self, texts, show_progress=False):
                import numpy as np
                return np.zeros((len(texts), 128))

        mock_emb = MockEmbeddingManager()
        retriever = HybridRetriever(embedding_manager=mock_emb, embedding_dim=128)

        assert retriever.embedding_manager is mock_emb
        assert retriever.count() == 0  # Empty by default

    def test_rrf_fusion_with_mock_embeddings(self):
        """Test RRF fusion logic with controlled embedding scores."""
        class MockEmbeddingManager:
            dimension = 128
            model_name = "mock"

            def embed_text(self, text):
                import numpy as np
                # Return deterministic vectors
                vec = np.zeros(128)
                if "physics" in text.lower():
                    vec[0] = 1.0
                elif "chemistry" in text.lower():
                    vec[1] = 1.0
                return vec

            def embed_batch(self, texts, show_progress=False):
                import numpy as np
                return np.array([self.embed_text(t) for t in texts])

        mock_emb = MockEmbeddingManager()
        retriever = HybridRetriever(embedding_manager=mock_emb, embedding_dim=128)

        chunks = [
            {"text": "Physics deals with matter", "metadata": {"source": "p.pdf", "page": 1}},
            {"text": "Chemistry is about substances", "metadata": {"source": "c.pdf", "page": 1}},
            {"text": "Biology studies living things", "metadata": {"source": "b.pdf", "page": 1}},
        ]

        retriever.index_documents(chunks)
        results = retriever.search("physics research", top_k=2)

        # "physics" embedding should match physics chunk
        assert results[0].text.startswith("Physics")