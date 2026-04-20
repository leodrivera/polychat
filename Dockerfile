# syntax=docker/dockerfile:1.7

# ---------- Build stage ----------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------- Runtime stage ----------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH \
    HF_HOME=/data/.hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/data/.hf_cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

RUN groupadd --system --gid 1001 polychat \
 && useradd --system --uid 1001 --gid polychat --home-dir /home/polychat --create-home polychat \
 && mkdir -p /app /data \
 && chown -R polychat:polychat /app /data

WORKDIR /app
COPY --from=builder --chown=polychat:polychat /app/.venv /app/.venv
COPY --chown=polychat:polychat src/ /app/src/
COPY --chown=polychat:polychat .streamlit/ /app/.streamlit/

USER polychat

# Persisted indexes and HF model cache live under /data — mount a volume here.
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status == 200 else 1)"

CMD ["streamlit", "run", "/app/src/polychat/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
