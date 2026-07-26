.PHONY: install dev test lint format api ui neo4j

install:
	python -m pip install -e ".[dev]"

dev:
	uvicorn tracerag.api.main:app --reload

api:
	uvicorn tracerag.api.main:app --host 0.0.0.0 --port 8000

ui:
	streamlit run src/tracerag/ui/app.py

neo4j:
	docker compose up -d neo4j

test:
	pytest

lint:
	ruff check .
	ruff format --check .
	mypy src

format:
	ruff check --fix .
	ruff format .
