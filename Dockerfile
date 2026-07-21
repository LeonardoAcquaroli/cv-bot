# ---------- Stage 1: build the React frontend ----------
FROM node:22-slim AS frontend
WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: Python backend + static SPA ----------
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install Python dependencies first (better layer caching).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code.
COPY backend/ ./backend/
COPY avatar.webp ./avatar.webp

# Compiled frontend from stage 1.
COPY --from=frontend /frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["sh", "-c", "uv run --no-dev uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
