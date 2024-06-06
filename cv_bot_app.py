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

st.title("Chat with Leo👋🏻")

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

avatar_image_url = 'https://raw.githubusercontent.com/LeonardoAcquaroli/cv-bot/main/avatar.webp'

#Openai Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Set the relative path to the docs directory in the repository
current_dir = os.path.dirname(os.path.abspath(__file__))
doc_folder = os.path.join(current_dir, 'docs')

# Initialize the ChatOpenAI instance with your model and API key
llm = ChatOpenAI(model=MODEL, temperature=0.000001, api_key=OPENAI_API_KEY)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar = avatar_image_url if message["role"] == "assistant" else None):
        st.markdown(message["content"])

@st.cache_resource
def display_chat_message(role, content):
    '''
    Display chat message in the chat message container and add it to chat history through session state.
    '''
    # Display user message in chat message container
    with st.chat_message(role):
        st.markdown(content)
    # Add user message to chat history
    st.session_state.messages.append({"role": role, "content": content})

# React to user input
if prompt := st.chat_input(placeholder='Ask about Leonardo'):

    display_chat_message("user", prompt)

    # Chatbot response
    response, search_result = handle_user_query(query=prompt,
                                            embeddings=embeddings,
                                            llm=llm)
    
    display_chat_message("assistant", response)