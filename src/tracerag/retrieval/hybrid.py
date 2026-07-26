import re

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from tracerag.domain import SearchResult
from tracerag.retrieval.index import LocalVectorIndex


class HybridRetriever:
    def __init__(self, index: LocalVectorIndex, rrf_k: int = 60) -> None:
        self.index = index
        self.rrf_k = rrf_k
        self._chunks = index.chunks
        self._bm25 = BM25Okapi([_tokenize(chunk.text) for chunk in self._chunks])

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        dense = self.index.search(query, limit=max(limit * 2, 10))
        sparse_scores = self._bm25.get_scores(_tokenize(query))
        sparse_indices = sorted(
            range(len(sparse_scores)), key=lambda index: sparse_scores[index], reverse=True
        )[: max(limit * 2, 10)]

        dense_ranks = {chunk.id: rank for rank, (chunk, _) in enumerate(dense, start=1)}
        sparse_ranks = {
            self._chunks[index].id: rank for rank, index in enumerate(sparse_indices, start=1)
        }
        chunks_by_id = {chunk.id: chunk for chunk, _ in dense}
        chunks_by_id.update(
            {self._chunks[index].id: self._chunks[index] for index in sparse_indices}
        )

        results: list[SearchResult] = []
        for chunk_id, chunk in chunks_by_id.items():
            dense_rank = dense_ranks.get(chunk_id)
            sparse_rank = sparse_ranks.get(chunk_id)
            score = 0.0
            if dense_rank is not None:
                score += 1 / (self.rrf_k + dense_rank)
            if sparse_rank is not None:
                score += 1 / (self.rrf_k + sparse_rank)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    dense_rank=dense_rank,
                    sparse_rank=sparse_rank,
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        return _normalize_scores(results[:limit])


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize_scores(results: list[SearchResult]) -> list[SearchResult]:
    if not results:
        return []
    maximum = results[0].score
    return [
        SearchResult(
            chunk=result.chunk,
            score=result.score / maximum if maximum else 0.0,
            dense_rank=result.dense_rank,
            sparse_rank=result.sparse_rank,
        )
        for result in results
    ]
