# secrets
import os
from dotenv import load_dotenv
# text
from langchain_openai import OpenAIEmbeddings
# chat
from rag import handle_user_query
from langchain_openai import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain.chains.question_answering import load_qa_chain
# Streamlit
import streamlit as st

load_dotenv()

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

#Openai Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Set the relative path to the docs directory in the repository
current_dir = os.path.dirname(os.path.abspath(__file__))
doc_folder = os.path.join(current_dir, 'docs')

# User query
query = st.chat_input(placeholder='Ask about Leonardo')

# Initialize the ChatOpenAI instance with your model and API key
llm = ChatOpenAI(model=MODEL, temperature=0.000001, api_key=OPENAI_API_KEY)

if query:
    
    response, search_result = handle_user_query(query=query,
                                                embeddings=embeddings,
                                                llm=llm)

    with st.chat_message(name='user'):
        st.write(response)