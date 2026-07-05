# RAG hybrid-search service image.
# One image serves both the API (default CMD) and the Streamlit UI (compose
# overrides the command). Heavy optional stacks (sentence-transformers/torch)
# are deliberately NOT installed — the default providers are API-based.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so code edits don't bust the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -e ".[api,ui,ingestion,indexing,llm]"

# Then the rest of the project (scripts, ui, eval, sample corpus).
COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
