# Chat with Leo

A RAG chatbot that answers questions about Leonardo Acquaroli's CV, skills, and
experience. Retrieval runs on Qdrant; answers are generated (and streamed) by
OpenAI. The UI is a React + TypeScript single-page app served by a FastAPI
backend as a single-process monolith.

## Architecture

```
frontend/  React + TypeScript (Vite)  ──►  backend/  FastAPI
                                              ├─ POST /api/chat   (SSE token stream)
                                              ├─ GET  /api/health
                                              └─ serves frontend/dist (SPA)
                                                   │
                                          Qdrant (vector search) + OpenAI (GPT-5 nano)
```

The RAG logic lives in [backend/rag.py](backend/rag.py) and prompts in
[backend/prompts.py](backend/prompts.py). The frontend consumes the SSE stream
and renders assistant replies token-by-token.

## Requirements

- Python `>=3.14` with [uv](https://docs.astral.sh/uv/)
- Node.js `>=18`
- A `.env` file (see below)

### Environment variables

```
OPENAI_API_KEY=...
QDRANT_API_URL=...
QDRANT_API_KEY=...
OPENAI_CHAT_MODEL=gpt-5-nano-2025-08-07   # optional
QDRANT_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2   # optional
QDRANT_COLLECTION=leo-docs                # optional
CVBOT_LOG_LEVEL=INFO                       # optional
```

## Local development

Run the backend and the Vite dev server in two terminals. The dev server
proxies `/api` to the backend.

```bash
# Terminal 1 — backend (http://localhost:8000)
uv sync
uv run uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend (http://localhost:5173)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Production (single process)

Build the frontend, then run the backend — it serves the compiled SPA:

```bash
cd frontend && npm install && npm run build && cd ..
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000.

## Docker

```bash
docker build -t cv-bot .
docker run --rm -p 8000:8000 --env-file .env cv-bot
```

## Tests

```bash
uv run pytest              # backend
cd frontend && npm test    # frontend
```
