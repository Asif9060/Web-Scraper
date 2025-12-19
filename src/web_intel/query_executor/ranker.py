"""
Result ranking and fusion for hybrid retrieval.

Combines results from multiple retrieval methods
using various fusion strategies.
"""

from dataclasses import dataclass, field
from enum import Enum

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class FusionMethod(str, Enum):
    """Method for fusing results from multiple sources."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"  # Weighted score combination
    MAX = "max"  # Maximum score across sources
    INTERLEAVE = "interleave"  # Round-robin interleaving


@dataclass
class RankedResult:
    """
    A result with ranking information.

    Contains the content and scores from different sources.
    """

    chunk_id: int
    page_id: int
    text: str
    final_score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    graph_score: float | None = None
    source_url: str = ""
    source_title: str = ""
    rank: int = 0

    def __hash__(self) -> int:
        return hash(self.chunk_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RankedResult):
            return False
        return self.chunk_id == other.chunk_id


class ResultRanker:
    """
    Ranks and fuses results from multiple retrieval sources.

    Combines vector search, keyword search, and graph-based
    retrieval results into a unified ranking.

    Example:
        >>> ranker = ResultRanker(method=FusionMethod.RRF)
        >>> ranked = ranker.fuse(
        ...     vector_results=vector_hits,
        ...     keyword_results=keyword_hits
        ... )
        >>> for result in ranked[:10]:
        ...     print(f"{result.rank}. {result.text[:50]} (score: {result.final_score:.3f})")
    """

    def __init__(
        self,
        method: FusionMethod = FusionMethod.RRF,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.3,
        graph_weight: float = 0.1,
        rrf_k: int = 60,
    ) -> None:
        """
        Initialize result ranker.

        Args:
            method: Fusion method to use
            vector_weight: Weight for vector search results
            keyword_weight: Weight for keyword search results
            graph_weight: Weight for graph-based results
            rrf_k: Constant k for RRF formula
        """
        self.method = method
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.graph_weight = graph_weight
        self.rrf_k = rrf_k

        # Normalize weights
        total = vector_weight + keyword_weight + graph_weight
        if total > 0:
            self.vector_weight /= total
            self.keyword_weight /= total
            self.graph_weight /= total

        logger.debug(f"ResultRanker initialized (method={method.value})")

    def fuse(
        self,
        vector_results: list[dict] | None = None,
        keyword_results: list[dict] | None = None,
        graph_results: list[dict] | None = None,
    ) -> list[RankedResult]:
        """
        Fuse results from multiple sources.

        Args:
            vector_results: Results from vector search
                Each dict should have: chunk_id, page_id, text, score, source_url, source_title
            keyword_results: Results from keyword search
            graph_results: Results from graph-based retrieval

        Returns:
            List of ranked results
        """
        vector_results = vector_results or []
        keyword_results = keyword_results or []
        graph_results = graph_results or []

        if self.method == FusionMethod.RRF:
            return self._fuse_rrf(vector_results, keyword_results, graph_results)
        elif self.method == FusionMethod.WEIGHTED:
            return self._fuse_weighted(vector_results, keyword_results, graph_results)
        elif self.method == FusionMethod.MAX:
            return self._fuse_max(vector_results, keyword_results, graph_results)
        elif self.method == FusionMethod.INTERLEAVE:
            return self._fuse_interleave(vector_results, keyword_results, graph_results)
        else:
            return self._fuse_rrf(vector_results, keyword_results, graph_results)

    def _fuse_rrf(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        graph_results: list[dict],
    ) -> list[RankedResult]:
        """
        Reciprocal Rank Fusion.

        RRF(d) = Σ 1 / (k + rank(d))
        """
        scores: dict[int, dict] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (self.rrf_k + rank)

            if chunk_id not in scores:
                scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "page_id": result.get("page_id", 0),
                    "text": result.get("text", ""),
                    "source_url": result.get("source_url", ""),
                    "source_title": result.get("source_title", ""),
                    "rrf_score": 0.0,
                    "vector_score": None,
                    "keyword_score": None,
                    "graph_score": None,
                }

            scores[chunk_id]["rrf_score"] += rrf_score * self.vector_weight
            scores[chunk_id]["vector_score"] = result.get("score")

        # Process keyword results
        for rank, result in enumerate(keyword_results, 1):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (self.rrf_k + rank)

            if chunk_id not in scores:
                scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "page_id": result.get("page_id", 0),
                    "text": result.get("text", ""),
                    "source_url": result.get("source_url", ""),
                    "source_title": result.get("source_title", ""),
                    "rrf_score": 0.0,
                    "vector_score": None,
                    "keyword_score": None,
                    "graph_score": None,
                }

            scores[chunk_id]["rrf_score"] += rrf_score * self.keyword_weight
            scores[chunk_id]["keyword_score"] = result.get("score")

        # Process graph results
        for rank, result in enumerate(graph_results, 1):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (self.rrf_k + rank)

            if chunk_id not in scores:
                scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "page_id": result.get("page_id", 0),
                    "text": result.get("text", ""),
                    "source_url": result.get("source_url", ""),
                    "source_title": result.get("source_title", ""),
                    "rrf_score": 0.0,
                    "vector_score": None,
                    "keyword_score": None,
                    "graph_score": None,
                }

            scores[chunk_id]["rrf_score"] += rrf_score * self.graph_weight
            scores[chunk_id]["graph_score"] = result.get("score")

        # Sort by RRF score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # Build ranked results
        ranked = []
        for rank, item in enumerate(sorted_results, 1):
            ranked.append(
                RankedResult(
                    chunk_id=item["chunk_id"],
                    page_id=item["page_id"],
                    text=item["text"],
                    final_score=item["rrf_score"],
                    vector_score=item["vector_score"],
                    keyword_score=item["keyword_score"],
                    graph_score=item["graph_score"],
                    source_url=item["source_url"],
                    source_title=item["source_title"],
                    rank=rank,
                )
            )

        return ranked

    def _fuse_weighted(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        graph_results: list[dict],
    ) -> list[RankedResult]:
        """Weighted score combination."""
        scores: dict[int, dict] = {}

        # Normalize scores within each source
        def normalize_scores(results: list[dict]) -> list[dict]:
            if not results:
                return results
            max_score = max(r.get("score", 0) for r in results) or 1.0
            for r in results:
                r["normalized_score"] = r.get("score", 0) / max_score
            return results

        vector_results = normalize_scores(vector_results)
        keyword_results = normalize_scores(keyword_results)
        graph_results = normalize_scores(graph_results)

        # Combine scores
        for result in vector_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = self._init_score_dict(result)
            scores[chunk_id]["weighted_score"] += (
                result.get("normalized_score", 0) * self.vector_weight
            )
            scores[chunk_id]["vector_score"] = result.get("score")

        for result in keyword_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = self._init_score_dict(result)
            scores[chunk_id]["weighted_score"] += (
                result.get("normalized_score", 0) * self.keyword_weight
            )
            scores[chunk_id]["keyword_score"] = result.get("score")

        for result in graph_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = self._init_score_dict(result)
            scores[chunk_id]["weighted_score"] += (
                result.get("normalized_score", 0) * self.graph_weight
            )
            scores[chunk_id]["graph_score"] = result.get("score")

        return self._build_ranked_results(scores, "weighted_score")

    def _fuse_max(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        graph_results: list[dict],
    ) -> list[RankedResult]:
        """Maximum score fusion."""
        scores: dict[int, dict] = {}

        for result in vector_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = self._init_score_dict(result)
            score = result.get("score", 0) * self.vector_weight
            scores[chunk_id]["max_score"] = max(
                scores[chunk_id].get("max_score", 0), score)
            scores[chunk_id]["vector_score"] = result.get("score")

        for result in keyword_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = self._init_score_dict(result)
            score = result.get("score", 0) * self.keyword_weight
            scores[chunk_id]["max_score"] = max(
                scores[chunk_id].get("max_score", 0), score)
            scores[chunk_id]["keyword_score"] = result.get("score")

        for result in graph_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = self._init_score_dict(result)
            score = result.get("score", 0) * self.graph_weight
            scores[chunk_id]["max_score"] = max(
                scores[chunk_id].get("max_score", 0), score)
            scores[chunk_id]["graph_score"] = result.get("score")

        return self._build_ranked_results(scores, "max_score")

    def _fuse_interleave(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        graph_results: list[dict],
    ) -> list[RankedResult]:
        """Round-robin interleaving."""
        seen = set()
        interleaved = []

        # Create iterators
        iters = [iter(vector_results), iter(
            keyword_results), iter(graph_results)]
        source_names = ["vector", "keyword", "graph"]

        while any(iters):
            for i, it in enumerate(iters):
                if it is None:
                    continue
                try:
                    result = next(it)
                    chunk_id = result["chunk_id"]
                    if chunk_id not in seen:
                        seen.add(chunk_id)
                        interleaved.append((result, source_names[i]))
                except StopIteration:
                    iters[i] = None

        # Build results with position-based scores
        ranked = []
        for rank, (result, source) in enumerate(interleaved, 1):
            ranked.append(
                RankedResult(
                    chunk_id=result["chunk_id"],
                    page_id=result.get("page_id", 0),
                    text=result.get("text", ""),
                    final_score=1.0 / rank,  # Position-based score
                    vector_score=result.get(
                        "score") if source == "vector" else None,
                    keyword_score=result.get(
                        "score") if source == "keyword" else None,
                    graph_score=result.get(
                        "score") if source == "graph" else None,
                    source_url=result.get("source_url", ""),
                    source_title=result.get("source_title", ""),
                    rank=rank,
                )
            )

        return ranked

    def _init_score_dict(self, result: dict) -> dict:
        """Initialize a score tracking dictionary."""
        return {
            "chunk_id": result["chunk_id"],
            "page_id": result.get("page_id", 0),
            "text": result.get("text", ""),
            "source_url": result.get("source_url", ""),
            "source_title": result.get("source_title", ""),
            "weighted_score": 0.0,
            "max_score": 0.0,
            "vector_score": None,
            "keyword_score": None,
            "graph_score": None,
        }

    def _build_ranked_results(
        self,
        scores: dict[int, dict],
        score_key: str,
    ) -> list[RankedResult]:
        """Build ranked results from score dictionary."""
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x.get(score_key, 0),
            reverse=True,
        )

        ranked = []
        for rank, item in enumerate(sorted_results, 1):
            ranked.append(
                RankedResult(
                    chunk_id=item["chunk_id"],
                    page_id=item["page_id"],
                    text=item["text"],
                    final_score=item.get(score_key, 0),
                    vector_score=item["vector_score"],
                    keyword_score=item["keyword_score"],
                    graph_score=item["graph_score"],
                    source_url=item["source_url"],
                    source_title=item["source_title"],
                    rank=rank,
                )
            )

        return ranked

    def rerank_with_diversity(
        self,
        results: list[RankedResult],
        diversity_weight: float = 0.3,
    ) -> list[RankedResult]:
        """
        Re-rank results to increase source diversity.

        Penalizes results from the same page to ensure
        diverse source coverage.

        Args:
            results: Initial ranked results
            diversity_weight: How much to penalize same-page results

        Returns:
            Re-ranked results
        """
        if not results:
            return results

        page_counts: dict[int, int] = {}
        reranked = []

        for result in results:
            page_id = result.page_id
            count = page_counts.get(page_id, 0)

            # Apply diversity penalty
            penalty = count * diversity_weight
            adjusted_score = result.final_score * (1.0 - penalty)

            reranked.append(
                RankedResult(
                    chunk_id=result.chunk_id,
                    page_id=result.page_id,
                    text=result.text,
                    final_score=adjusted_score,
                    vector_score=result.vector_score,
                    keyword_score=result.keyword_score,
                    graph_score=result.graph_score,
                    source_url=result.source_url,
                    source_title=result.source_title,
                    rank=0,
                )
            )

            page_counts[page_id] = count + 1

        # Re-sort and assign ranks
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        for i, result in enumerate(reranked, 1):
            result.rank = i

        return reranked
