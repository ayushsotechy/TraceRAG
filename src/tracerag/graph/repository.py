from collections import defaultdict
from typing import Protocol

from tracerag.domain import Chunk, EntityMention, GraphRelationship


class GraphRepository(Protocol):
    def initialize(self) -> None: ...

    def replace(
        self,
        chunks: list[Chunk],
        mentions: list[EntityMention],
        relationships: list[GraphRelationship],
    ) -> None: ...

    def search_chunk_ids(self, query_terms: list[str], limit: int) -> list[str]: ...

    def close(self) -> None: ...


class InMemoryGraphRepository:
    def __init__(self) -> None:
        self._entity_chunks: dict[str, set[str]] = defaultdict(set)
        self._neighbors: dict[str, set[str]] = defaultdict(set)

    def initialize(self) -> None:
        return

    def replace(
        self,
        chunks: list[Chunk],
        mentions: list[EntityMention],
        relationships: list[GraphRelationship],
    ) -> None:
        del chunks
        self._entity_chunks.clear()
        self._neighbors.clear()
        for mention in mentions:
            self._entity_chunks[mention.entity.normalized_name].add(mention.chunk_id)
        for relationship in relationships:
            source = relationship.source.normalized_name
            target = relationship.target.normalized_name
            self._neighbors[source].add(target)
            self._neighbors[target].add(source)

    def search_chunk_ids(self, query_terms: list[str], limit: int) -> list[str]:
        scores: dict[str, int] = defaultdict(int)
        matched_entities = {
            entity
            for entity in self._entity_chunks
            if any(term in entity or entity in term for term in query_terms)
        }
        expanded = matched_entities | {
            neighbor for entity in matched_entities for neighbor in self._neighbors[entity]
        }
        for entity in expanded:
            weight = 2 if entity in matched_entities else 1
            for chunk_id in self._entity_chunks[entity]:
                scores[chunk_id] += weight
        return [
            chunk_id
            for chunk_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
                :limit
            ]
        ]

    def close(self) -> None:
        return


class Neo4jGraphRepository:
    def __init__(self, uri: str, username: str, password: str, database: str) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Install the Neo4j Python driver") from exc
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    def initialize(self) -> None:
        self._driver.verify_connectivity()
        self._driver.execute_query(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            database_=self._database,
        )
        self._driver.execute_query(
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.normalized_name IS UNIQUE",
            database_=self._database,
        )

    def replace(
        self,
        chunks: list[Chunk],
        mentions: list[EntityMention],
        relationships: list[GraphRelationship],
    ) -> None:
        self._driver.execute_query("MATCH (n) DETACH DELETE n", database_=self._database)
        chunk_rows = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page": chunk.page,
            }
            for chunk in chunks
        ]
        mention_rows = [
            {
                "chunk_id": mention.chunk_id,
                "name": mention.entity.name,
                "normalized_name": mention.entity.normalized_name,
                "entity_type": mention.entity.entity_type,
            }
            for mention in mentions
        ]
        relationship_rows = [
            {
                "source": relationship.source.normalized_name,
                "target": relationship.target.normalized_name,
                "chunk_id": relationship.chunk_id,
                "kind": relationship.relationship_type,
            }
            for relationship in relationships
        ]
        self._driver.execute_query(
            """
            UNWIND $rows AS row
            CREATE (:Chunk {
                id: row.id, document_id: row.document_id,
                filename: row.filename, page: row.page
            })
            """,
            rows=chunk_rows,
            database_=self._database,
        )
        self._driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (c:Chunk {id: row.chunk_id})
            MERGE (e:Entity {normalized_name: row.normalized_name})
            SET e.name = row.name, e.entity_type = row.entity_type
            MERGE (c)-[:MENTIONS]->(e)
            """,
            rows=mention_rows,
            database_=self._database,
        )
        self._driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (source:Entity {normalized_name: row.source})
            MATCH (target:Entity {normalized_name: row.target})
            MERGE (source)-[r:RELATED_TO {chunk_id: row.chunk_id}]->(target)
            SET r.kind = row.kind
            """,
            rows=relationship_rows,
            database_=self._database,
        )

    def search_chunk_ids(self, query_terms: list[str], limit: int) -> list[str]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (matched:Entity)
            WHERE any(term IN $terms WHERE
                matched.normalized_name CONTAINS term OR term CONTAINS matched.normalized_name)
            OPTIONAL MATCH (matched)-[:RELATED_TO]-(neighbor:Entity)
            WITH collect(DISTINCT matched) + collect(DISTINCT neighbor) AS entities
            UNWIND entities AS entity
            MATCH (chunk:Chunk)-[:MENTIONS]->(entity)
            RETURN chunk.id AS chunk_id, count(*) AS graph_score
            ORDER BY graph_score DESC
            LIMIT $limit
            """,
            terms=query_terms,
            limit=limit,
            database_=self._database,
        )
        return [str(record["chunk_id"]) for record in records]

    def close(self) -> None:
        self._driver.close()
