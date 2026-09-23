FROM ghcr.io/astral-sh/uv:0.12.6 AS uv
FROM python:3.12.11-slim
ARG APP_VERSION=dev
LABEL org.opencontainers.image.revision=${APP_VERSION}
ENV APP_VERSION=${APP_VERSION} PATH=/app/.venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
COPY --from=uv /uv /bin/uv
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app app && mkdir -p /app /data/documents && chown app:app /app
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
USER app
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
