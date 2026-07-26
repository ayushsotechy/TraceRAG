import hashlib
from pathlib import Path

from pypdf import PdfReader

from tracerag.domain import Document


class UnsupportedDocumentError(ValueError):
    pass


def load_document(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        return [_document(path, text, page=1)]
    raise UnsupportedDocumentError(f"Unsupported document type: {suffix}")


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(path)
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(_document(path, text, page=page_number))
    return documents


def _document(path: Path, text: str, page: int) -> Document:
    digest = hashlib.sha256(f"{path.name}:{page}:{text}".encode()).hexdigest()[:20]
    return Document(id=digest, filename=path.name, text=text, page=page)
