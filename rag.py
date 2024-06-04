# Chat
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
# Embeddings
from langchain_openai import OpenAIEmbeddings
# Vector DB
from pymongo.mongo_client import MongoClient
# Environment
import os
from dotenv import load_dotenv
load_dotenv()

# divide into two classes one for vector search and query and the other for mongodb connection to collection

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

class MongoDB:
    def __init__(self, cluster_url='la19.fjkkeei.mongodb.net') -> None:
        # Credentials from environment variables 
        self.username = os.getenv('MONGO_CLUSTER_USER')
        self.password = os.getenv('MONGO_CLUSTER_PASS')
        self.app_name = os.getenv('APP_NAME')
        self.cluster_url = cluster_url
        self.uri = f"mongodb+srv://{self.username}:{self.password}@{self.cluster_url}/?retryWrites=true&w=majority&appName={self.app_name}"
    
    def get_client(self):
        # Create a new clientand connect to server
        mongo_client = MongoClient(self.uri)

        # Ping to check connection
        try:
            mongo_client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
            return mongo_client
        except Exception as e:
            print('There was an issue onnecting to MongoDB.\n')
            print(e)
        
    def get_db(self, db_name='cv-bot'):
        mongo_client = self.get_client()
        db = mongo_client[db_name]
        return db
    
    def get_collection(self, collection_name='my_documents'):
        db = self.get_db()
        collection = db[collection_name]
        return collection

def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embeddings = embeddings.embed_query(user_query)

    if query_embeddings is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "textual_docs_vector_index",
                "queryVector": query_embeddings,
                "path": "embeddings",
                "numCandidates": 100,  # Number of candidate matches to consider
                "limit": 2  # Return top 5 matches
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

def handle_user_query(query):
    # Get connection to MongoDB only in the moment of a prompt from the webapp
    collection = MongoDB.get_collection() # It gets instantiated every time a message comes...FIX

    get_knowledge = vector_search(query, collection)

    search_result = ''
    for result in get_knowledge:
        search_result += f'''Document title: {result.get('doc_title', 'N/A')},
                             Document text: {result.get('text', 'N/A')}\n'''

    # Initialize the ChatOpenAI instance with your model and API key
    llm = ChatOpenAI(model=MODEL, temperature=0.0001, api_key=OPENAI_API_KEY)

    # Prepare the system prompt
    system_prompt = '''You are a hiring assistant built by Leonardo Acquaroli and you have to give recruiters information about Leonardo that can convince them to hire him without showing too much that you are biased towards him.
        Provide detailed answers mostly based on the passed context.'''

    # Prepare the user prompt with the query and search results
    user_prompt = f'''Answer this user query: "{query}", with the following context: "{search_result}"'''

    # Create the ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template(user_prompt)
        ]
    )
    
    # Prepare the input for the invoke method
    formatted_input = prompt_template.format(input={"query": query, "context": search_result})

    # Invoke the LLM with the formatted input
    response = llm.invoke(input=formatted_input)
    
    return response.content, search_result