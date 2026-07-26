import shutil
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from tracerag import __version__
from tracerag.api.schemas import (
    AnswerResponse,
    CitationResponse,
    HealthResponse,
    IngestResponse,
    QuestionRequest,
)
from tracerag.config import get_settings
from tracerag.ingestion.loaders import UnsupportedDocumentError
from tracerag.service import RAGService

app = FastAPI(
    title="TraceRAG API",
    description="Evaluation-first retrieval-augmented generation with source citations.",
    version=__version__,
)


@lru_cache
def get_service() -> RAGService:
    return RAGService(get_settings())


ServiceDependency = Annotated[RAGService, Depends(get_service)]


@app.get("/health", response_model=HealthResponse)
def health(service: ServiceDependency) -> HealthResponse:
    return HealthResponse(
        status="ok",
        documents=service.document_count,
        chunks=service.chunk_count,
        retrieval_mode=service.retrieval_mode,
    )


@app.post("/v1/documents", response_model=IngestResponse)
def ingest_documents(
    files: Annotated[list[UploadFile], File()],
    service: ServiceDependency,
) -> IngestResponse:
    upload_dir = service.settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        for upload in files:
            safe_name = Path(upload.filename or "document").name
            path = upload_dir / safe_name
            with path.open("wb") as destination:
                shutil.copyfileobj(upload.file, destination)
            paths.append(path)
        chunks = service.ingest(paths)
        return IngestResponse(files=len(paths), chunks=chunks)
    except (UnsupportedDocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/query", response_model=AnswerResponse)
def query(
    request: QuestionRequest,
    service: ServiceDependency,
) -> AnswerResponse:
    result = service.ask(request.question)
    return AnswerResponse(
        answer=result.text,
        citations=[CitationResponse(**asdict(citation)) for citation in result.citations],
        confidence=result.confidence,
        abstained=result.abstained,
    )
