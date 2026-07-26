from types import SimpleNamespace

from tracerag.domain import Chunk, SearchResult
from tracerag.generation.generator import OpenAIGroundedGenerator
from tracerag.graph.extraction import (
    ExtractedEntity,
    ExtractedRelationship,
    GraphExtraction,
    OpenAIEntityExtractor,
)


class FakeResponses:
    def create(self, **kwargs: object) -> SimpleNamespace:
        assert "Evidence:" in str(kwargs["input"])
        return SimpleNamespace(output_text="Hybrid retrieval combines both methods. [1]")

    def parse(self, **kwargs: object) -> SimpleNamespace:
        assert kwargs["text_format"] is GraphExtraction
        return SimpleNamespace(
            output_parsed=GraphExtraction(
                entities=[
                    ExtractedEntity(name="Hybrid Retrieval"),
                    ExtractedEntity(name="BM25", entity_type="METHOD"),
                ],
                relationships=[
                    ExtractedRelationship(
                        source="Hybrid Retrieval",
                        target="BM25",
                        relationship_type="USES",
                    )
                ],
            )
        )


class FakeClient:
    responses = FakeResponses()


def test_openai_generator_returns_grounded_citations() -> None:
    result = SearchResult(
        Chunk("c1", "d1", "guide.pdf", 4, "Hybrid retrieval combines BM25 and vectors."),
        0.9,
    )
    generator = OpenAIGroundedGenerator("test", "test-model", client=FakeClient())

    answer = generator.generate("What is hybrid retrieval?", [result])

    assert not answer.abstained
    assert answer.citations[0].page == 4
    assert "[1]" in answer.text


def test_openai_extractor_creates_typed_relationships() -> None:
    extractor = OpenAIEntityExtractor("test", "test-model", client=FakeClient())
    chunk = Chunk("c1", "d1", "guide.pdf", 1, "Hybrid Retrieval uses BM25.")

    mentions, relationships = extractor.extract([chunk])

    assert len(mentions) == 2
    assert relationships[0].relationship_type == "USES"
    assert relationships[0].source.normalized_name == "hybrid retrieval"
