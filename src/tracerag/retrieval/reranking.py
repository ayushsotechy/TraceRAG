from typing import Protocol

import numpy as np

from tracerag.domain import SearchResult


class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], limit: int) -> list[SearchResult]: ...


class NoOpReranker:
    def rerank(self, query: str, results: list[SearchResult], limit: int) -> list[SearchResult]:
        del query
        return results[:limit]


class CrossEncoderReranker:
    """Multilingual cross-encoder reranker for query-passage relevance."""

    def __init__(self, model_name: str, model: object | None = None) -> None:
        if model is not None:
            self._model = model
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install TraceRAG with the 'ml' extra") from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], limit: int) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, result.chunk.text) for result in results]
        raw_scores = np.asarray(self._model.predict(pairs), dtype=np.float32)  # type: ignore[attr-defined]
        probabilities = 1.0 / (1.0 + np.exp(-raw_scores))
        ranked = sorted(
            zip(results, probabilities, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            SearchResult(
                chunk=result.chunk,
                score=float(score),
                dense_rank=result.dense_rank,
                sparse_rank=result.sparse_rank,
            )
            for result, score in ranked[:limit]
        ]
