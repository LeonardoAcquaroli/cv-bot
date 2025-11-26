"""RAG utilities powered by Cohere embeddings + OpenAI GPT-5 Nano."""

import logging
import os

import cohere
from dotenv import load_dotenv
from openai import OpenAI
from pymongo.mongo_client import MongoClient

from prompts import ChatPrompts  # system_prompt, user_prompt

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano-2025-08-07")
EMBEDDING_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")
LOG_LEVEL = os.getenv("CVBOT_LOG_LEVEL", "INFO").upper()

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set.")

if not COHERE_API_KEY:
    raise EnvironmentError("COHERE_API_KEY is not set.")

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
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

class MongoDB:
    def __init__(self, cluster_url='la19.fjkkeei.mongodb.net', db_name=None) -> None:
        self.username = os.getenv('MONGO_CLUSTER_USER')
        self.password = os.getenv('MONGO_CLUSTER_PASS')
        self.app_name = os.getenv('APP_NAME')
        self.cluster_url = cluster_url
        self.db_name = db_name or os.getenv('MONGO_DB_NAME', 'my_database')
        self.uri = f"mongodb+srv://{self.username}:{self.password}@{self.cluster_url}/?retryWrites=true&w=majority&appName={self.app_name}"
    
    def get_client(self):
        mongo_client = MongoClient(self.uri)
        try:
            mongo_client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
            return mongo_client
        except Exception as e:
            print('There was an issue connecting to MongoDB.\n')
            print(e)
        
    def get_db(self):
        mongo_client = self.get_client()
        return mongo_client[self.db_name]
    
    def get_collection(self, collection_name='my_documents'):
        db = self.get_db()
        return db[collection_name]

def _extract_embedding_vector(embed_response) -> list:
    """Return the default float embedding vector from Cohere's embed response."""
    embeddings = getattr(embed_response, "embeddings", None)
    if embeddings is None:
        raise ValueError("Embed response missing embeddings")
    if hasattr(embeddings, "float_"):
        vector = embeddings.float_[0]
    elif hasattr(embeddings, "float"):
        vector = embeddings.float[0]
    else:
        vector = embeddings[0]
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return vector


def embed_query_text(text: str) -> list:
    response = cohere_client.embed(
        texts=[text],
        model=EMBEDDING_MODEL,
        input_type="search_query",
    )
    vector = _extract_embedding_vector(response)
    logger.debug("Generated query embedding (%d dims) for text snippet: %s", len(vector), text[:60])
    return vector


def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    logger.info("Running vector search for query: %s", user_query)
    query_embeddings = embed_query_text(user_query)

    if query_embeddings is None:
        logger.warning("Embedding generation failed for query: %s", user_query)
        return []

    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "textual_docs_vector_index",
                "queryVector": query_embeddings,
                "path": "embeddings",
                "numCandidates": 20,  # Number of candidate matches to consider
                "limit": 3  # Return top 3 matches
            }
        },
        {
            "$project": {
                "_id": 0,  # Exclude the _id field
                "doc_title": 1,
                "text": 1,
                "score": {
                    "$meta": "vectorSearchScore"  # Include the search score
                }
            }
        }
    ]

    try:
        results = list(collection.aggregate(pipeline))
    except Exception as exc:
        logger.exception("Vector search failed: %s", exc)
        return []

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


def handle_user_query(query, collection, chat_history=None):
    logger.info("Handling user query '%s' (chat history len=%d)", query, len(chat_history or []))
    knowledge = vector_search(query, collection)

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

    prompts = ChatPrompts()
    history_text = "\n".join(chat_history or [])
    context_for_prompt = search_result or "No relevant context retrieved from MongoDB for this query."

    user_prompt = prompts.user_prompt.format(
        query=query,
        search_result=context_for_prompt,
        chat_history=history_text,
    )

    response = openai_client.chat.completions.create(
        model=MODEL,
        temperature=1,
        messages=[
            {"role": "system", "content": prompts.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    logger.info("LLM response generated for query: %s", query)
    return response.choices[0].message.content, search_result