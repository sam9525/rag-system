"""Tests for hybrid retriever."""

import pytest
import numpy as np
from src.hybrid_retriever import HybridRetriever, RRFResult


class TestHybridRetriever:
    """Test HybridRetriever class."""

    def test_rrf_fusion_ranks_correctly(self):
        """Test RRF fusion combines rankings properly."""
        retriever = HybridRetriever(embedding_dim=128)

        # Two sets of rankings - now using (chunk_index, score)
        semantic_results = [
            (0, 0.9),  # doc_a at index 0
            (1, 0.8),  # doc_b at index 1
            (2, 0.7),  # doc_c at index 2
        ]
        keyword_results = [
            (1, 1.0),  # doc_b at index 1
            (2, 0.9),  # doc_c at index 2
            (0, 0.6),  # doc_a at index 0
        ]

        fused = retriever._rrf_fusion(semantic_results, keyword_results, k=60)

        # doc_b (index 1) should be ranked first (top in both)
        assert fused[0][0] == 1  # index 1

    @pytest.mark.skip(reason="Requires full integration test with real embeddings")
    def test_search_returns_final_top_k(self):
        """Test search returns final top k chunks."""
        retriever = HybridRetriever(embedding_dim=128)
        pass

    def test_empty_corpus_handling(self):
        """Test handling of empty corpus."""
        retriever = HybridRetriever(embedding_dim=128)
        results = retriever.search("query", top_k=3)
        assert len(results) == 0

    def test_inject_mock_embedding_manager(self):
        """Test HybridRetriever accepts injected embedding manager."""
        class MockEmbeddingManager:
            @property
            def dimension(self):
                return 128

            @property
            def model_name(self):
                return "mock"

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
            @property
            def dimension(self):
                return 128

            @property
            def model_name(self):
                return "mock"

            def embed_text(self, text):
                import numpy as np
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
        assert results[0].metadata.get("source") == "p.pdf"

    def test_retriever_with_mock_embedding_manager(self):
        """Test HybridRetriever works with mock EmbeddingManager."""
        class MockEmbeddingManager:
            dimension = 128
            model_name = "mock"

            def embed_text(self, text):
                import numpy as np
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

        # Index some chunks
        chunks = [
            {"text": "Physics deals with matter and energy", "metadata": {"source": "p.pdf", "page": 1}},
            {"text": "Chemistry is about substances and reactions", "metadata": {"source": "c.pdf", "page": 1}},
        ]

        retriever.index_documents(chunks)
        results = retriever.search("physics research", top_k=2)

        # Should return results
        assert len(results) >= 1
        assert results[0].text.startswith("Physics")

    def test_auto_infer_dimension_from_manager(self):
        """Test that HybridRetriever infers dimension from injected manager."""
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
        # Should NOT need embedding_dim parameter when manager is provided
        retriever = HybridRetriever(embedding_manager=mock_emb)

        # Vector store should have the right dimension
        assert retriever.vector_store.dimension == 128
        assert retriever._embedding_dim == 128

    def test_load_chunks_from_external_source(self):
        """Test loading chunks from external source (e.g., chunks.json)."""
        retriever = HybridRetriever(embedding_dim=128)

        chunks = [
            {"text": "First chunk", "metadata": {"source": "doc.pdf", "page": 1}},
            {"text": "Second chunk", "metadata": {"source": "doc.pdf", "page": 2}},
        ]

        # Load chunks directly
        retriever.load_chunks(chunks)

        assert retriever.count() == 2
        # Chunks should be available for search results
        # (note: vector_store is empty, so search won't work without indexing)