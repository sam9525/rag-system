"""Reciprocal Rank Fusion for combining ranked retrieval results."""

from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class RRFResult:
    """Result from RRF fusion with full chunk data."""

    text: str
    score: float
    metadata: dict
    chunk_index: int = 0
    semantic_score: float | None = None
    keyword_score: float | None = None


def rrf_fusion(
    semantic_results: List[Tuple[int, float]],
    keyword_results: List[Tuple[int, float]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Apply Reciprocal Rank Fusion.

    RRF_score(d) = Σ 1/(k + rank_i(d))

    Args:
        semantic_results: List of (chunk_index, score) from semantic search
        keyword_results: List of (chunk_index, score) from keyword search
        k: RRF smoothing parameter (default 60)

    Returns:
        Combined ranking as list of (chunk_index, rrf_score)
    """
    rrf_scores: Dict[int, float] = {}

    for rank, (chunk_idx, _) in enumerate(semantic_results, 1):
        if chunk_idx not in rrf_scores:
            rrf_scores[chunk_idx] = 0
        rrf_scores[chunk_idx] += 1 / (k + rank)

    for rank, (chunk_idx, _) in enumerate(keyword_results, 1):
        if chunk_idx not in rrf_scores:
            rrf_scores[chunk_idx] = 0
        rrf_scores[chunk_idx] += 1 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
