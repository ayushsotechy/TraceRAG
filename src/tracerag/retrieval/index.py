import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tracerag.domain import Chunk
from tracerag.retrieval.embeddings import Embedder


class LocalVectorIndex:
    """Small persistent cosine-similarity index with explicit, portable storage."""

    def __init__(self, directory: Path, embedder: Embedder) -> None:
        self.directory = directory
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._embeddings = np.empty((0, 0), dtype=np.float32)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        self._persist()

    def load(self) -> bool:
        metadata_path = self.directory / "chunks.json"
        vectors_path = self.directory / "vectors.npy"
        if not metadata_path.exists() or not vectors_path.exists():
            return False
        raw_chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk(**item) for item in raw_chunks]
        self._embeddings = np.load(vectors_path)
        return True

    def search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []
        query_vector = self.embedder.embed([query])[0]
        scores: NDArray[np.float32] = self._embeddings @ query_vector
        indices = np.argsort(scores)[::-1][:limit]
        return [(self._chunks[index], float(scores[index])) for index in indices]

    def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        requested = set(chunk_ids)
        chunks_by_id = {chunk.id: chunk for chunk in self._chunks if chunk.id in requested}
        return [chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id]

    def _persist(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        chunks = [asdict(chunk) for chunk in self._chunks]
        (self.directory / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        np.save(self.directory / "vectors.npy", self._embeddings)
