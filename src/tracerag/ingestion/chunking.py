import hashlib
import re

from tracerag.domain import Chunk, Document


def chunk_documents(
    documents: list[Document], chunk_size: int = 900, overlap: int = 150
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size")

    chunks: list[Chunk] = []
    for document in documents:
        for index, text in enumerate(_split_text(document.text, chunk_size, overlap)):
            chunk_id = hashlib.sha256(f"{document.id}:{index}:{text}".encode()).hexdigest()[:20]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    filename=document.filename,
                    page=document.page,
                    text=text,
                    metadata={"chunk_index": str(index)},
                )
            )
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = max(normalized.rfind(". ", start, end), normalized.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = end - overlap
    return chunks
