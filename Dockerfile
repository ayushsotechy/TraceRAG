FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "tracerag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

