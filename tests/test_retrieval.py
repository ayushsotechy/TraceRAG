from pathlib import Path

from tracerag.domain import Chunk
from tracerag.retrieval.embeddings import HashEmbedder
from tracerag.retrieval.hybrid import HybridRetriever
from tracerag.retrieval.index import LocalVectorIndex


def test_hybrid_retrieval_returns_relevant_chunk(tmp_path: Path) -> None:
    chunks = [
        Chunk("1", "d1", "python.md", 1, "Python uses indentation to define code blocks."),
        Chunk("2", "d2", "db.md", 1, "PostgreSQL supports transactional database workloads."),
        Chunk("3", "d3", "rag.md", 1, "Hybrid retrieval combines semantic search and BM25."),
    ]
    index = LocalVectorIndex(tmp_path, HashEmbedder())
    index.build(chunks)

    results = HybridRetriever(index).search("How does hybrid retrieval use BM25?", limit=2)

    assert results[0].chunk.id == "3"
    assert results[0].score == 1.0
