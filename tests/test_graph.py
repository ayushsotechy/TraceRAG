from pathlib import Path

from tracerag.domain import Chunk
from tracerag.graph.extraction import LocalEntityExtractor
from tracerag.graph.repository import InMemoryGraphRepository
from tracerag.retrieval.embeddings import HashEmbedder
from tracerag.retrieval.graph_hybrid import GraphHybridRetriever
from tracerag.retrieval.hybrid import HybridRetriever
from tracerag.retrieval.index import LocalVectorIndex


def test_extractor_builds_mentions_and_relationships() -> None:
    chunks = [
        Chunk(
            "chunk-1",
            "doc-1",
            "guide.md",
            1,
            "Hybrid Retrieval combines BM25 with Vector Search.",
        )
    ]

    mentions, relationships = LocalEntityExtractor().extract(chunks)

    names = {mention.entity.normalized_name for mention in mentions}
    assert "bm25" in names
    assert relationships
    assert all(relationship.chunk_id == "chunk-1" for relationship in relationships)


def test_graph_hybrid_promotes_graph_connected_chunk(tmp_path: Path) -> None:
    chunks = [
        Chunk(
            "fusion",
            "doc-1",
            "guide.md",
            1,
            "Reciprocal Rank Fusion combines result lists.",
        ),
        Chunk(
            "bm25",
            "doc-1",
            "guide.md",
            2,
            "BM25 performs sparse keyword retrieval.",
        ),
        Chunk("other", "doc-2", "other.md", 1, "Docker packages applications."),
    ]
    extractor = LocalEntityExtractor()
    mentions, relationships = extractor.extract(chunks)
    graph = InMemoryGraphRepository()
    graph.replace(chunks, mentions, relationships)
    index = LocalVectorIndex(tmp_path, HashEmbedder())
    index.build(chunks)
    retriever = GraphHybridRetriever(HybridRetriever(index), graph, extractor)

    results = retriever.search("How is BM25 used?", limit=3)

    assert results[0].chunk.id == "bm25"
    assert results[0].score == 1.0
