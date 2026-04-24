# syntax=docker/dockerfile:1.7


# Builder — resolve deps, bake the embedding model + DuckDB index into the layer
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HOME=/app/.hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache

WORKDIR /app

# Deps first for layer-cache friendliness
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Pre-bake sentence-transformers/all-MiniLM-L6-v2 (~80MB) so the runtime
# image is fully offline — no first-run Hugging Face Hub dependency.
RUN uv run python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Pre-build the DuckDB index from the commit-pinned awesome-mcp-servers README.
# Shifts the ~30s index build + outbound fetch from first-run to build-time,
# and makes the resulting image deterministic.
RUN uv run python -m mcpilot.indexer


# Runtime — distroless-ish slim base, non-root, offline-by-default
FROM python:3.12-slim-bookworm AS runtime

ARG VERSION=0.1.1
LABEL org.opencontainers.image.title="mcpilot" \
      org.opencontainers.image.description="Context-aware MCP server advisor — recommends MCP servers for your project and explains why." \
      org.opencontainers.image.source="https://github.com/yahiaklk/mcpilot" \
      org.opencontainers.image.documentation="https://github.com/yahiaklk/mcpilot#readme" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}"

RUN groupadd --system --gid 10001 mcpilot && \
    useradd  --system --uid 10001 --gid mcpilot --home /app --shell /sbin/nologin mcpilot

WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

COPY --from=builder --chown=mcpilot:mcpilot /app /app

USER mcpilot

# FastMCP default transport is stdio — correct for local MCP clients
# (Claude Desktop, Claude Code, etc) wiring via `docker run -i`.
# For remote hosted deployment (e.g. Glama inspector) once server.py accepts
# a --transport flag, override CMD to: ["--transport", "sse", "--port", "8000"]
ENTRYPOINT ["python", "-m", "mcpilot.server"]
