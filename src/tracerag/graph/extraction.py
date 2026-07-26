import itertools
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from tracerag.domain import Chunk, Entity, EntityMention, GraphRelationship

_STOPWORDS = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "Every",
    "For",
    "From",
    "If",
    "In",
    "It",
    "Of",
    "On",
    "Or",
    "The",
    "This",
    "To",
    "When",
    "With",
}


class EntityExtractor(Protocol):
    def extract(
        self, chunks: list[Chunk]
    ) -> tuple[list[EntityMention], list[GraphRelationship]]: ...

    def query_entities(self, text: str) -> list[str]: ...


class LocalEntityExtractor:
    """Deterministic technical-entity extraction baseline."""

    def extract(self, chunks: list[Chunk]) -> tuple[list[EntityMention], list[GraphRelationship]]:
        mentions: list[EntityMention] = []
        relationships: list[GraphRelationship] = []
        for chunk in chunks:
            entities = self._entities(chunk.text)
            mentions.extend(EntityMention(entity=entity, chunk_id=chunk.id) for entity in entities)
            relationships.extend(
                GraphRelationship(
                    source=source,
                    target=target,
                    relationship_type="CO_OCCURS_WITH",
                    chunk_id=chunk.id,
                )
                for source, target in itertools.combinations(entities, 2)
            )
        return mentions, relationships

    def query_entities(self, text: str) -> list[str]:
        extracted = [entity.normalized_name for entity in self._entities(text)]
        tokens = re.findall(r"[a-z0-9][a-z0-9-]{2,}", text.lower())
        return list(dict.fromkeys([*extracted, *tokens]))[:12]

    def _entities(self, text: str) -> list[Entity]:
        candidates = re.findall(
            r"\b(?:[A-Z][A-Za-z0-9+#.-]{1,}(?:\s+[A-Z][A-Za-z0-9+#.-]{1,}){0,3}|[A-Z]{2,})\b",
            text,
        )
        entities: dict[str, Entity] = {}
        for candidate in candidates:
            name = candidate.strip(" .")
            if name in _STOPWORDS or len(name) < 2:
                continue
            normalized = re.sub(r"\s+", " ", name.lower())
            entities[normalized] = Entity(
                name=name,
                normalized_name=normalized,
                entity_type="ACRONYM" if name.isupper() else "CONCEPT",
            )
        return list(entities.values())[:20]


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str = "CONCEPT"


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    relationship_type: str


class GraphExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class OpenAIEntityExtractor:
    """Extracts typed entities and relationships with structured outputs."""

    def __init__(
        self,
        api_key: str,
        model: str,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model
        self._local = LocalEntityExtractor()

    def extract(self, chunks: list[Chunk]) -> tuple[list[EntityMention], list[GraphRelationship]]:
        mentions: list[EntityMention] = []
        relationships: list[GraphRelationship] = []
        for chunk in chunks:
            response = self._client.responses.parse(
                model=self._model,
                instructions=(
                    "Extract canonical entities and meaningful directed relationships "
                    "from the text. Use concise uppercase relationship types such as "
                    "USES, COMBINES, PART_OF, CREATED_BY, or DEPENDS_ON. Do not invent facts."
                ),
                input=chunk.text,
                text_format=GraphExtraction,
            )
            parsed = response.output_parsed
            if parsed is None:
                continue
            entities = {
                item.name.lower().strip(): Entity(
                    name=item.name.strip(),
                    normalized_name=item.name.lower().strip(),
                    entity_type=item.entity_type.upper(),
                )
                for item in parsed.entities
                if item.name.strip()
            }
            mentions.extend(
                EntityMention(entity=entity, chunk_id=chunk.id) for entity in entities.values()
            )
            for item in parsed.relationships:
                source = entities.get(item.source.lower().strip())
                target = entities.get(item.target.lower().strip())
                if source is not None and target is not None:
                    relationships.append(
                        GraphRelationship(
                            source=source,
                            target=target,
                            relationship_type=item.relationship_type.upper(),
                            chunk_id=chunk.id,
                        )
                    )
        return mentions, relationships

    def query_entities(self, text: str) -> list[str]:
        return self._local.query_entities(text)
