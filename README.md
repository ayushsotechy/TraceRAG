# TraceRAG

TraceRAG is an evaluation-first retrieval-augmented generation application for technical
documents. It combines dense and BM25 retrieval using reciprocal-rank fusion, returns
page-level citations, and abstains when evidence is insufficient.

It also includes an optional GraphRAG mode backed by Neo4j. GraphRAG extracts technical
entities, connects entities that occur in the same source chunk, traverses neighboring
entities for a question, and fuses graph-ranked evidence with the text retrieval ranks.

## Architecture

```text
PDF / Markdown / text
        |
        v
loader -> metadata-aware chunker -> embedding index
                                      + BM25 index
                                           |
                                           v
                               reciprocal-rank fusion
                                           |
                                           v
                             grounded answer + citations
```

The default providers are deliberately credential-free:

- `HashEmbedder` gives deterministic local dense retrieval.
- `ExtractiveGenerator` creates evidence-only answers.

Provider interfaces allow the local baseline to be replaced with Sentence Transformers
and an LLM without coupling those dependencies to the retrieval domain.

For multilingual semantic retrieval and reranking:

```env
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RERANKER_PROVIDER=cross-encoder
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

Install the ML dependencies with `python -m pip install -e ".[ml]"`. The configured
models support cross-language retrieval and multilingual query-passage reranking.

For grounded OpenAI answers and typed knowledge-graph extraction:

```env
GENERATION_PROVIDER=openai
ENTITY_EXTRACTION_PROVIDER=openai
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4.1-mini
```

The generator receives only retrieved, numbered evidence, requires inline citations,
answers in the question's language, and abstains when evidence is insufficient. Graph
extraction uses structured outputs to create canonical entities and meaningful typed
relationships such as `USES`, `COMBINES`, `PART_OF`, and `DEPENDS_ON`.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Run the API:

```bash
make api
```

In another terminal, run the UI:

```bash
make ui
```

Open `http://localhost:8501`, upload one or more documents, build the index, and ask a
question. API documentation is available at `http://localhost:8000/docs`.

## Enable GraphRAG

Start the local Neo4j Community Edition container:

```bash
make neo4j
```

Update `.env`:

```env
RETRIEVAL_MODE=graph_hybrid
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=tracerag-password
NEO4J_DATABASE=neo4j
```

Restart the API, then upload the documents again. Ingestion will create:

```text
(Chunk)-[:MENTIONS]->(Entity)
(Entity)-[:RELATED_TO {kind: "CO_OCCURS_WITH"}]->(Entity)
```

Neo4j Browser is available at `http://localhost:7474`. The default local credentials
match the values above. Use a unique secret outside local development.

To run the complete containerized application:

```bash
cp .env.example .env
docker compose up --build
```

`hybrid` mode remains the default and does not require a running Neo4j database when the
API is started directly with `make api`.

## Deploy on Render

The included `render.yaml` deploys a public Streamlit UI and FastAPI service on free plans.
Both use the credential-free hybrid retrieval and extractive generation defaults, so no
API keys or Neo4j instance are required.

1. In the Render dashboard, choose **New > Blueprint**.
2. Connect this repository and select the `main` branch.
3. Review the two services and apply the Blueprint.
4. Open the URL for `tracerag-ui`, upload a document, build the index, and ask a question.

Uploaded files and the index use ephemeral storage in this demo configuration. They may
be cleared whenever the API service restarts or redeploys, so upload the demo document
again when needed. Add a persistent disk mounted at `/app/data` if long-lived indexes are
required.

## API example

```bash
curl -F "files=@document.pdf" http://localhost:8000/v1/documents
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document recommend?"}'
```

## Quality checks

```bash
make test
make lint
```

## Retrieval evaluation

TraceRAG includes an offline evaluation runner for reproducible Recall@K and mean
reciprocal rank measurements. Each evaluation case contains a question and one or more
relevant chunk IDs:

```json
[
  {
    "question": "How does the system combine retrieval results?",
    "relevant_chunk_ids": ["the-id-from-data-index-chunks-json"]
  }
]
```

Use `load_cases`, `evaluate_retrieval`, and `save_report` from
`tracerag.evaluation.runner` to create a versioned evaluation report. A later milestone
will add a CLI and comparison reports for dense-only versus hybrid retrieval.

## Roadmap

- Language-aware chunking
- Retrieval and faithfulness evaluation dataset
- Incremental document ingestion and deletion
- OpenTelemetry traces and request metrics
- Authentication, rate limiting, and cloud deployment
