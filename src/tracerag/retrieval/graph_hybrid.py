from tracerag.domain import SearchResult
from tracerag.graph.extraction import EntityExtractor
from tracerag.graph.repository import GraphRepository
from tracerag.retrieval.hybrid import HybridRetriever


class GraphHybridRetriever:
    """Fuses hybrid text ranks with graph-neighborhood ranks using RRF."""

    def __init__(
        self,
        text_retriever: HybridRetriever,
        graph_repository: GraphRepository,
        extractor: EntityExtractor,
        rrf_k: int = 60,
    ) -> None:
        self.text_retriever = text_retriever
        self.graph_repository = graph_repository
        self.extractor = extractor
        self.rrf_k = rrf_k

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        text_results = self.text_retriever.search(query, limit=max(limit * 2, 10))
        graph_ids = self.graph_repository.search_chunk_ids(
            self.extractor.query_entities(query),
            limit=max(limit * 2, 10),
        )
        text_by_id = {result.chunk.id: result for result in text_results}
        for chunk in self.text_retriever.index.get_by_ids(graph_ids):
            text_by_id.setdefault(chunk.id, SearchResult(chunk=chunk, score=0.0))
        text_ranks = {result.chunk.id: rank for rank, result in enumerate(text_results, start=1)}
        graph_ranks = {chunk_id: rank for rank, chunk_id in enumerate(graph_ids, start=1)}

        fused: list[SearchResult] = []
        for chunk_id, result in text_by_id.items():
            score = 1 / (self.rrf_k + text_ranks[chunk_id]) if chunk_id in text_ranks else 0.0
            if chunk_id in graph_ranks:
                score += 1 / (self.rrf_k + graph_ranks[chunk_id])
            fused.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    dense_rank=result.dense_rank,
                    sparse_rank=result.sparse_rank,
                )
            )
        fused.sort(key=lambda item: item.score, reverse=True)
        if not fused:
            return []
        maximum = fused[0].score
        return [
            SearchResult(
                chunk=result.chunk,
                score=result.score / maximum,
                dense_rank=result.dense_rank,
                sparse_rank=result.sparse_rank,
            )
            for result in fused[:limit]
        ]
