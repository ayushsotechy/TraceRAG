from tracerag.domain import Chunk, SearchResult
from tracerag.retrieval.reranking import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        assert len(pairs) == 2
        return [-2.0, 3.0]


def test_cross_encoder_reranks_by_predicted_relevance() -> None:
    results = [
        SearchResult(Chunk("1", "d", "a.md", 1, "less relevant"), 0.9),
        SearchResult(Chunk("2", "d", "a.md", 2, "more relevant"), 0.5),
    ]
    reranker = CrossEncoderReranker("unused", model=FakeCrossEncoder())

    reranked = reranker.rerank("question", results, limit=2)

    assert [result.chunk.id for result in reranked] == ["2", "1"]
    assert reranked[0].score > reranked[1].score
