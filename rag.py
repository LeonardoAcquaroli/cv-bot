"""RAG utilities powered by Cohere embeddings + OpenAI GPT-5 Nano."""

from prompts import ChatPrompts  # system_prompt, user_prompt
from pymongo.mongo_client import MongoClient
from openai import OpenAI
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano-2025-08-07")
EMBEDDING_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set.")

if not COHERE_API_KEY:
    raise EnvironmentError("COHERE_API_KEY is not set.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

class MongoDB:
    def __init__(self, cluster_url='la19.fjkkeei.mongodb.net') -> None:
        # Credentials from environment variables 
        self.username = os.getenv('MONGO_CLUSTER_USER')
        self.password = os.getenv('MONGO_CLUSTER_PASS')
        self.app_name = os.getenv('APP_NAME')
        self.cluster_url = cluster_url
        self.uri = f"mongodb+srv://{self.username}:{self.password}@{self.cluster_url}/?retryWrites=true&w=majority&appName={self.app_name}"
    
    def get_client(self):
        # Create a new client and connect to server
        mongo_client = MongoClient(self.uri)

        # Ping to check connection
        try:
            mongo_client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
            return mongo_client
        except Exception as e:
            print('There was an issue connecting to MongoDB.\n')
            print(e)
        
    def get_db(self, db_name='cv-bot'):
        mongo_client = self.get_client()
        db = mongo_client[db_name]
        return db
    
    def get_collection(self, collection_name='my_documents'):
        db = self.get_db()
        collection = db[collection_name]
        return collection

def _extract_embedding_vector(embed_response) -> list:
    """Return the default float embedding vector from Cohere's embed response."""
    embeddings = getattr(embed_response, "embeddings", None)
    if embeddings is None:
        raise ValueError("Embed response missing embeddings")
    if hasattr(embeddings, "float_"):
        return embeddings.float_[0]
    return embeddings[0]


def embed_query_text(text: str) -> list:
    response = cohere_client.embed(
        texts=[text],
        model=EMBEDDING_MODEL,
        input_type="search_query",
    )
    return _extract_embedding_vector(response)


def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    query_embeddings = embed_query_text(user_query)

    if query_embeddings is None:
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
                "doc_title": 1,  # Include the doc_title field
                "text": 1,  # Include the text field
                "embeddings": 1,
                "score": {
                    "$meta": "vectorSearchScore"  # Include the search score
                }
            }
        }
    ]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)


def handle_user_query(query, collection, chat_history=None):
    knowledge = vector_search(query, collection)

    search_result = ''
    for result in knowledge:
        search_result += (
            f"Document title: {result.get('doc_title', 'N/A')}, "
            f"Document text: {result.get('text', 'N/A')}"
        )

    prompts = ChatPrompts()
    history_text = "\n".join(chat_history or [])
    user_prompt = prompts.user_prompt.format(
        query=query,
        search_result=search_result,
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

    return response.choices[0].message.content, search_result