FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[service]"

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8080

CMD ["uvicorn", "industrial_defect.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
