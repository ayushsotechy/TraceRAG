import hashlib
import re
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> NDArray[np.float32]: ...


class HashEmbedder:
    """Deterministic feature-hashing embedder for local development and tests."""

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                value = int.from_bytes(digest)
                column = value % self.dimensions
                sign = 1.0 if value & 1 else -1.0
                matrix[row, column] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.asarray(matrix / np.maximum(norms, 1e-12), dtype=np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install TraceRAG with the 'ml' extra") from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        result = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(result, dtype=np.float32)
