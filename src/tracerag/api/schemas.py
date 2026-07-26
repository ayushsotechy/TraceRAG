from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class CitationResponse(BaseModel):
    source_id: int
    filename: str
    page: int
    excerpt: str


class AnswerResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    confidence: float
    abstained: bool


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    retrieval_mode: str


class IngestResponse(BaseModel):
    files: int
    chunks: int
