"""Fusion strategies for combining retrieval results."""

from src.retrieval.search_result import SearchResult
from src.retrieval.retrieval_engine import FusionStrategy


class RRFFusion(FusionStrategy):
    """Reciprocal Rank Fusion for combining ranked retrieval results.

    RRF_score(d) = Σ 1/(k + rank_i(d))
    """

    def __init__(self, k: int = 60):
        self._k = k

    def fuse(
        self, results: list[list[SearchResult]], k: int | None = None
    ) -> list[SearchResult]:
        """Fuse results from multiple engines using RRF."""
        k = k or self._k
        rrf_scores: dict[int, float] = {}

        for engine_results in results:
            for rank, result in enumerate(engine_results, 1):
                idx = result.chunk_index
                if idx not in rrf_scores:
                    rrf_scores[idx] = 0.0
                rrf_scores[idx] += 1.0 / (k + rank)

        # Reconstruct SearchResult objects with fused scores
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for chunk_idx, rrf_score in ranked:
            # Get text/metadata from the first result list that has this chunk
            for engine_results in results:
                for r in engine_results:
                    if r.chunk_index == chunk_idx:
                        fused.append(
                            SearchResult(
                                chunk_index=chunk_idx,
                                score=rrf_score,
                                text=r.text,
                                metadata=r.metadata,
                            )
                        )
                        break
        return fused
