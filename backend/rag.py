"""RAG utilities powered by Qdrant (fastembed inference) + OpenAI GPT-5 Nano."""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Document

from backend.prompts import USER_PROMPT, SYSTEM_PROMPT

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano-2025-08-07")
EMBEDDING_MODEL = os.getenv("QDRANT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "leo-docs")
LOG_LEVEL = os.getenv("CVBOT_LOG_LEVEL", "INFO").upper()

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set.")

if not QDRANT_API_URL:
    raise EnvironmentError("QDRANT_API_URL is not set.")

if not QDRANT_API_KEY:
    raise EnvironmentError("QDRANT_API_KEY is not set.")

logging_levels = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

logger = logging.getLogger("cv_bot.rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging_levels.get(LOG_LEVEL, logging.INFO))
logger.propagate = False

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_API_URL, api_key=QDRANT_API_KEY, cloud_inference=True)


qdrant_client = get_qdrant_client()


def vector_search(user_query, client=None, collection_name=COLLECTION_NAME, limit=5):
    """
    Perform a vector search in the Qdrant collection based on the user query.

    Args:
    user_query (str): The user's query string.
    client (QdrantClient): Qdrant client to search with; defaults to the module client.
    collection_name (str): Collection to query.
    limit (int): Number of matches to return.

    Returns:
    list: A list of dicts with doc_title, text and score.
    """

    logger.info("Running vector search for query: %s", user_query)
    client = client or qdrant_client

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=Document(text=user_query, model=EMBEDDING_MODEL),
            with_payload=True,
            limit=limit,
        )
    except Exception as exc:
        logger.exception("Vector search failed: %s", exc)
        return []

    results = [
        {
            "doc_title": point.payload.get("doc_name", "N/A"),
            "text": point.payload.get("text", ""),
            "score": point.score,
        }
        for point in response.points
    ]

    if results:
        top = results[0]
        logger.info(
            "Vector search matched %d docs. Top result '%s' (score %.4f)",
            len(results),
            top.get('doc_title', 'N/A'),
            top.get('score', 0.0)
        )
    else:
        logger.warning("Vector search returned no matches for query: %s", user_query)

    return results

def _build_context_and_messages(query, client=None, chat_history=None):
    """Run retrieval and assemble the (messages, search_result) pair shared by
    the blocking and streaming query handlers."""
    logger.info("Handling user query '%s' (chat history len=%d)", query, len(chat_history or []))
    knowledge = vector_search(query, client)

    context_chunks = []
    for result in knowledge:
        snippet = result.get('text', 'N/A')
        snippet = snippet.strip().replace('\n', ' ')
        context_chunks.append(
            f"Title: {result.get('doc_title', 'N/A')} | Score: {result.get('score', 0):.4f} | Text: {snippet[:600]}"
        )

    search_result = "\n".join(context_chunks)

    if not search_result:
        logger.warning("No retrieved context available for query: %s", query)
    else:
        logger.debug("Context passed to LLM (first 300 chars): %s", search_result[:300])

    history_text = "\n".join(chat_history or [])
    context_for_prompt = search_result or "No relevant context retrieved from Qdrant for this query."

    user_prompt = USER_PROMPT.format(
        query=query,
        search_result=context_for_prompt,
        chat_history=history_text,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, search_result


def handle_user_query(query, client=None, chat_history=None):
    messages, search_result = _build_context_and_messages(query, client, chat_history)

    response = openai_client.chat.completions.create(
        model=MODEL,
        temperature=1,
        messages=messages,
    )
    logger.info("LLM response generated for query: %s", query)
    return response.choices[0].message.content, search_result


def handle_user_query_stream(query, client=None, chat_history=None):
    """Yield the assistant answer as text chunks as OpenAI generates them.

    Mirrors ``handle_user_query`` but streams tokens. Raises on failure so the
    caller (API layer) can surface a graceful error event.
    """
    messages, _ = _build_context_and_messages(query, client, chat_history)

    stream = openai_client.chat.completions.create(
        model=MODEL,
        temperature=1,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            yield text
    logger.info("LLM streaming response completed for query: %s", query)
