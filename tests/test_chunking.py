import pytest

from tracerag.domain import Document
from tracerag.ingestion.chunking import chunk_documents


def test_chunking_preserves_source_metadata() -> None:
    document = Document(id="doc-1", filename="guide.pdf", text="Sentence. " * 100, page=3)
    chunks = chunk_documents([document], chunk_size=200, overlap=30)

    assert len(chunks) > 1
    assert all(chunk.filename == "guide.pdf" for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=200, overlap=200)
