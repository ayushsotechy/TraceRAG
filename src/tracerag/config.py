from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "TraceRAG"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("data")

    chunk_size: int = Field(default=900, ge=200, le=3000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    top_k: int = Field(default=8, ge=1, le=50)
    rerank_top_k: int = Field(default=5, ge=1, le=20)
    min_relevance_score: float = Field(default=0.15, ge=0, le=1)

    embedding_provider: Literal["hash", "sentence-transformers"] = "hash"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_provider: Literal["none", "cross-encoder"] = "none"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    generation_provider: Literal["extractive", "openai"] = "extractive"
    entity_extraction_provider: Literal["local", "openai"] = "local"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    retrieval_mode: Literal["hybrid", "graph_hybrid"] = "hybrid"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "tracerag-password"
    neo4j_database: str = "neo4j"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


@lru_cache
def get_settings() -> Settings:
    return Settings()
