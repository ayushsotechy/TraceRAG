import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tracerag.retrieval.hybrid import HybridRetriever


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    question: str
    relevant_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    cases: int
    recall_at_k: float
    mean_reciprocal_rank: float


def evaluate_retrieval(
    retriever: HybridRetriever,
    cases: list[EvaluationCase],
    k: int = 5,
) -> EvaluationReport:
    if not cases:
        raise ValueError("Evaluation set cannot be empty")

    hits = 0
    reciprocal_ranks = 0.0
    for case in cases:
        retrieved = retriever.search(case.question, limit=k)
        relevant = set(case.relevant_chunk_ids)
        ranks = [
            rank for rank, result in enumerate(retrieved, start=1) if result.chunk.id in relevant
        ]
        if ranks:
            hits += 1
            reciprocal_ranks += 1 / min(ranks)

    return EvaluationReport(
        cases=len(cases),
        recall_at_k=hits / len(cases),
        mean_reciprocal_rank=reciprocal_ranks / len(cases),
    )


def load_cases(path: Path) -> list[EvaluationCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvaluationCase(
            question=item["question"],
            relevant_chunk_ids=tuple(item["relevant_chunk_ids"]),
        )
        for item in payload
    ]


def save_report(report: EvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
