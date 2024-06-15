# Chat
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
from prompts import ChatPrompts #system_prompt, user_prompt

# Vector DB
from pymongo.mongo_client import MongoClient

# Environment
import os
from dotenv import load_dotenv
load_dotenv()

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

def vector_search(user_query, collection, embeddings):
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
                "numCandidates": 20,  # Number of candidate matches to consider
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

def handle_user_query(query, collection, embeddings, llm, chat_history):

    knowledge = vector_search(query, collection, embeddings)

    search_result = ''
    for result in knowledge:
        search_result += f'''Document title: {result.get('doc_title', 'N/A')},
                             Document text: {result.get('text', 'N/A')}'''

    # Prepare the system prompt
    system_prompt = ChatPrompts().system_prompt

    # Prepare the user prompt with the query and search results
    user_prompt = ChatPrompts().user_prompt.format(query=query, search_result=search_result, chat_history=chat_history)    

    # Create the ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template(user_prompt)
        ]
    )

    # Prepare the input for the chain
    chain = prompt_template | llm | RunnableLambda(lambda x: f'{x.content}')

    # Invoke the chain with the formatted input
    response = chain.invoke(input={
                'query': query, 
                'context': search_result
                })
    
    return response, search_result