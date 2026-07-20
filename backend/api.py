"""HTTP API for the CV bot: SSE chat streaming + health check."""

import json
import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.rag import get_qdrant_client, handle_user_query_stream

logger = logging.getLogger("cv_bot.api")

router = APIRouter()

# The Qdrant client is expensive to build (loads inference config), so create
# it once at import time and reuse it across requests.
_qdrant_client = get_qdrant_client()


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request):
    chat_history = [f"{turn.role}: {turn.content}" for turn in payload.history]

    async def event_generator():
        try:
            for chunk in handle_user_query_stream(
                query=payload.message,
                client=_qdrant_client,
                chat_history=chat_history,
            ):
                if await request.is_disconnected():
                    logger.info("Client disconnected mid-stream; aborting.")
                    break
                yield {"event": "token", "data": json.dumps({"text": chunk})}
        except Exception:  # noqa: BLE001 - surface a safe message, keep details in logs
            logger.exception("Chat streaming failed")
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": "Something went wrong while generating the answer. Please try again."}
                ),
            }
        finally:
            yield {"event": "done", "data": "{}"}

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            # Disable proxy buffering (e.g. nginx) so tokens flush immediately.
            "X-Accel-Buffering": "no",
        },
    )
