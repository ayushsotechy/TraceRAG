import re
from typing import Any, Protocol

from tracerag.domain import Answer, Citation, SearchResult


class Generator(Protocol):
    def generate(self, question: str, contexts: list[SearchResult]) -> Answer: ...


class ExtractiveGenerator:
    """Credential-free grounded baseline that selects evidence-bearing sentences."""

    def __init__(self, min_score: float = 0.15) -> None:
        self.min_score = min_score

    def generate(self, question: str, contexts: list[SearchResult]) -> Answer:
        if not contexts or contexts[0].score < self.min_score:
            return Answer(
                text="I don't have enough evidence in the indexed documents to answer that.",
                citations=(),
                confidence=0.0,
                abstained=True,
            )

        question_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        selected: list[tuple[str, SearchResult, float]] = []
        for result in contexts[:4]:
            sentences = re.split(r"(?<=[.!?])\s+", result.chunk.text)
            for sentence in sentences:
                terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                overlap = len(question_terms & terms) / max(len(question_terms), 1)
                selected.append((sentence, result, overlap))
        selected.sort(key=lambda item: (item[2], item[1].score), reverse=True)

        citations: list[Citation] = []
        answer_parts: list[str] = []
        seen_chunks: set[str] = set()
        for sentence, result, _ in selected:
            if result.chunk.id in seen_chunks or len(sentence) < 30:
                continue
            source_id = len(citations) + 1
            answer_parts.append(f"{sentence} [{source_id}]")
            citations.append(
                Citation(
                    source_id=source_id,
                    filename=result.chunk.filename,
                    page=result.chunk.page,
                    excerpt=result.chunk.text[:280],
                )
            )
            seen_chunks.add(result.chunk.id)
            if len(answer_parts) == 3:
                break

        confidence = min(1.0, contexts[0].score * 0.7 + (len(answer_parts) / 3) * 0.3)
        return Answer(
            text=" ".join(answer_parts),
            citations=tuple(citations),
            confidence=confidence,
            abstained=not answer_parts,
        )


class OpenAIGroundedGenerator:
    """Generates answers exclusively from numbered retrieved evidence."""

    def __init__(
        self,
        api_key: str,
        model: str,
        min_score: float = 0.15,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model
        self._min_score = min_score

    def generate(self, question: str, contexts: list[SearchResult]) -> Answer:
        if not contexts or contexts[0].score < self._min_score:
            return Answer(
                text="I don't have enough evidence in the indexed documents to answer that.",
                citations=(),
                confidence=0.0,
                abstained=True,
            )
        evidence = "\n\n".join(
            f"[{index}] Source: {result.chunk.filename}, page {result.chunk.page}\n"
            f"{result.chunk.text}"
            for index, result in enumerate(contexts, start=1)
        )
        response = self._client.responses.create(
            model=self._model,
            instructions=(
                "Answer only from the supplied evidence. Cite every factual claim using "
                "[n]. If the evidence cannot answer the question, return exactly "
                "INSUFFICIENT_EVIDENCE. Answer in the same language as the question."
            ),
            input=f"Question: {question}\n\nEvidence:\n{evidence}",
        )
        text = str(response.output_text).strip()
        if text == "INSUFFICIENT_EVIDENCE":
            return Answer(
                text="I don't have enough evidence in the indexed documents to answer that.",
                citations=(),
                confidence=0.0,
                abstained=True,
            )
        cited_ids = {int(value) for value in re.findall(r"\[(\d+)\]", text)}
        citations = tuple(
            Citation(
                source_id=index,
                filename=result.chunk.filename,
                page=result.chunk.page,
                excerpt=result.chunk.text[:280],
            )
            for index, result in enumerate(contexts, start=1)
            if index in cited_ids
        )
        confidence = min(1.0, contexts[0].score * 0.7 + (0.3 if citations else 0.0))
        return Answer(
            text=text,
            citations=citations,
            confidence=confidence,
            abstained=False,
        )
