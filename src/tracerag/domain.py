from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    filename: str
    text: str
    page: int


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    filename: str
    page: int
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass(frozen=True, slots=True)
class Citation:
    source_id: int
    filename: str
    page: int
    excerpt: str


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    confidence: float
    abstained: bool = False


@dataclass(frozen=True, slots=True)
class Entity:
    name: str
    normalized_name: str
    entity_type: str = "CONCEPT"


@dataclass(frozen=True, slots=True)
class EntityMention:
    entity: Entity
    chunk_id: str


@dataclass(frozen=True, slots=True)
class GraphRelationship:
    source: Entity
    target: Entity
    relationship_type: str
    chunk_id: str
