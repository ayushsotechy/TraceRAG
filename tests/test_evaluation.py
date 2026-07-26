from pathlib import Path

from tracerag.domain import Chunk
from tracerag.evaluation.runner import EvaluationCase, evaluate_retrieval
from tracerag.retrieval.embeddings import HashEmbedder
from tracerag.retrieval.hybrid import HybridRetriever
from tracerag.retrieval.index import LocalVectorIndex


def test_evaluation_computes_recall_and_mrr(tmp_path: Path) -> None:
    chunks = [
        Chunk("rag", "d1", "rag.md", 1, "RAG retrieves evidence before generation."),
        Chunk("sql", "d2", "sql.md", 1, "SQL is used to query relational databases."),
    ]
    index = LocalVectorIndex(tmp_path, HashEmbedder())
    index.build(chunks)
    retriever = HybridRetriever(index)

    report = evaluate_retrieval(
        retriever,
        [EvaluationCase("How does RAG use evidence?", ("rag",))],
        k=2,
    )

    assert report.recall_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
