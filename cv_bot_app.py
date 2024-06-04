# secrets
import os
from dotenv import load_dotenv
# text
from langchain_openai import OpenAIEmbeddings
# chat
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
# streamlit
import streamlit as st

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

#Openai Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Set the relative path to the docs directory in the repository
current_dir = os.path.dirname(os.path.abspath(__file__))
doc_folder = os.path.join(current_dir, 'docs')

# User query
query = st.chat_input(placeholder='Ask about Leonardo')

if query:
    embedding_vector = embeddings.embed_query(query)
    # Perform vector_search

    template = """
                Given the following extracted parts of a long document and a question, create a final answer only with the context information.

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot:"""

    llm = ChatOpenAI(temperature=0.0001,
                    model_name="gpt-3.5-turbo")

    prompt = PromptTemplate(input_variables=["chat_history", "human_input", "context"],
                            template=template)

    memory = ConversationBufferMemory(memory_key="chat_history",
                                    input_key="human_input")

    #stuff= i use all chunk retrived without any preprocessing
    chain = load_qa_chain(llm,
                        chain_type="stuff",
                        memory=memory,
                        prompt=prompt)
    
    docs = retriever.invoke(query)
    output = chain.invoke({"input_documents": docs, "human_input": query}, return_only_outputs=True)

    with st.chat_message(name='user'):
        st.write(output['output_text'])