from pathlib import Path

from tracerag.config import Settings
from tracerag.service import RAGService


def test_service_ingests_and_answers(tmp_path: Path) -> None:
    document = tmp_path / "rag.txt"
    document.write_text(
        "Reciprocal rank fusion combines ranked lists from dense and sparse retrieval.",
        encoding="utf-8",
    )
    settings = Settings(
        data_dir=tmp_path / "data",
        min_relevance_score=0,
        retrieval_mode="hybrid",
        generation_provider="extractive",
        entity_extraction_provider="local",
    )
    service = RAGService(settings)

    chunk_count = service.ingest([document])
    answer = service.ask("What does reciprocal rank fusion combine?")

    assert chunk_count == 1
    assert not answer.abstained
    assert answer.citations[0].filename == "rag.txt"
