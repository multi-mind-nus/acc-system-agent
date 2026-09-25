FROM ghcr.io/astral-sh/uv:0.12.18-python3.12-trixie-slim
ARG APP_VERSION=dev
LABEL org.opencontainers.image.revision=${APP_VERSION}
ENV APP_VERSION=${APP_VERSION} PATH=/app/.venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
# PDFium needs local substitutes for unembedded Latin and Chinese PDF fonts.
RUN apt-get update && apt-get install -y --no-install-recommends fonts-liberation fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app app && mkdir -p /app /data/documents && chown app:app /app
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
USER app
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
