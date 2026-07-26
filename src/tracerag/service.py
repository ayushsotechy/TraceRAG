from pathlib import Path
from typing import Protocol

from tracerag.config import Settings
from tracerag.domain import Answer, SearchResult
from tracerag.generation.generator import ExtractiveGenerator, OpenAIGroundedGenerator
from tracerag.graph.extraction import (
    EntityExtractor,
    LocalEntityExtractor,
    OpenAIEntityExtractor,
)
from tracerag.graph.repository import GraphRepository, Neo4jGraphRepository
from tracerag.ingestion.chunking import chunk_documents
from tracerag.ingestion.loaders import load_document
from tracerag.retrieval.embeddings import HashEmbedder, SentenceTransformerEmbedder
from tracerag.retrieval.graph_hybrid import GraphHybridRetriever
from tracerag.retrieval.hybrid import HybridRetriever
from tracerag.retrieval.index import LocalVectorIndex
from tracerag.retrieval.reranking import CrossEncoderReranker, NoOpReranker, Reranker


class Retriever(Protocol):
    def search(self, query: str, limit: int = 8) -> list[SearchResult]: ...


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        embedder = (
            SentenceTransformerEmbedder(settings.embedding_model)
            if settings.embedding_provider == "sentence-transformers"
            else HashEmbedder()
        )
        self.index = LocalVectorIndex(settings.index_dir, embedder)
        self.index.load()
        self._graph_repository: GraphRepository | None = None
        self._entity_extractor: EntityExtractor = (
            OpenAIEntityExtractor(
                api_key=settings.openai_api_key or "",
                model=settings.openai_model,
            )
            if settings.entity_extraction_provider == "openai"
            else LocalEntityExtractor()
        )
        if settings.retrieval_mode == "graph_hybrid":
            self._graph_repository = Neo4jGraphRepository(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
                database=settings.neo4j_database,
            )
            self._graph_repository.initialize()
        self._retriever = self._build_retriever() if self.index.chunks else None
        self._reranker: Reranker = (
            CrossEncoderReranker(settings.reranker_model)
            if settings.reranker_provider == "cross-encoder"
            else NoOpReranker()
        )
        self._generator = (
            OpenAIGroundedGenerator(
                api_key=settings.openai_api_key or "",
                model=settings.openai_model,
                min_score=settings.min_relevance_score,
            )
            if settings.generation_provider == "openai"
            else ExtractiveGenerator(settings.min_relevance_score)
        )

    @property
    def document_count(self) -> int:
        return len({chunk.document_id for chunk in self.index.chunks})

    @property
    def chunk_count(self) -> int:
        return len(self.index.chunks)

    @property
    def retrieval_mode(self) -> str:
        return self.settings.retrieval_mode

    def ingest(self, paths: list[Path]) -> int:
        documents = [document for path in paths for document in load_document(path)]
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        self.index.build(chunks)
        if self._graph_repository is not None:
            mentions, relationships = self._entity_extractor.extract(chunks)
            self._graph_repository.replace(chunks, mentions, relationships)
        self._retriever = self._build_retriever()
        return len(chunks)

    def ask(self, question: str) -> Answer:
        if self._retriever is None:
            return Answer(
                text="No documents have been indexed yet.",
                citations=(),
                confidence=0.0,
                abstained=True,
            )
        contexts = self._retriever.search(question, self.settings.top_k)
        reranked = self._reranker.rerank(question, contexts, limit=self.settings.rerank_top_k)
        return self._generator.generate(question, reranked)

    def close(self) -> None:
        if self._graph_repository is not None:
            self._graph_repository.close()

    def _build_retriever(self) -> Retriever:
        text_retriever = HybridRetriever(self.index)
        if self._graph_repository is None:
            return text_retriever
        return GraphHybridRetriever(
            text_retriever=text_retriever,
            graph_repository=self._graph_repository,
            extractor=self._entity_extractor,
        )
