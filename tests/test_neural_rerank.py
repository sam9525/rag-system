import pytest
from src.neural_rerank import NeuralRerank, RerankResult


class TestNeuralRerank:
    def test_rerank_initialization(self):
        rerank = NeuralRerank()
        assert rerank.model_name == "cross-encoder/ms-marco-MiniLM-L-12v2"
        assert rerank.default_model == "cross-encoder/ms-marco-MiniLM-L-12v2"

    def test_rerank_empty_chunks(self):
        rerank = NeuralRerank()
        query = "test query"
        results = rerank(query, [], top_k=3)
        assert len(results) == 0

    def test_rerank_with_insufficient_chunks(self):
        # Test with actual model (requires network/download)
        import pytest

        pytest.skip("Requires model download - tested manually")


class TestRerankResult:
    def test_rerank_result_dataclass(self):
        result = RerankResult(text="test", rerank_score=0.95, original_index=2)
        assert result.text == "test"
        assert result.rerank_score == 0.95
        assert result.original_index == 2


def test_config_has_reranking_options():
    from src.config import config

    assert hasattr(config.retrieval, "use_neural_rerank")
    assert hasattr(config.retrieval, "rerank_model")
    assert hasattr(config.retrieval, "rerank_top_k")


def test_hybrid_retriever_rerank_is_none_by_default():
    from src.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    assert retriever.rerank is None


def test_hybrid_retriever_with_rerank():
    from src.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    rerank = NeuralRerank()

    retriever.set_rerank(rerank)
    assert retriever.rerank is not None
    assert isinstance(retriever.rerank, NeuralRerank)


def test_rag_system_rerank_disabled_by_default():
    from src.config import config
    from src.rag_system import RAGSystem

    config.retrieval.use_neural_rerank = False
    rag = RAGSystem(source_dir=None)
    assert rag.retriever.rerank is None


def test_rag_system_rerank_enabled_via_config():
    from src.config import config
    from src.rag_system import RAGSystem

    config.retrieval.use_neural_rerank = True
    config.retrieval.rerank_model = "cross-encoder/ms-marco-MiniLM-L-12v2"

    rag = RAGSystem(source_dir=None)
    assert rag.retriever.rerank is not None
    assert isinstance(rag.retriever.rerank, NeuralRerank)

    # Reset config
    config.retrieval.use_neural_rerank = False
