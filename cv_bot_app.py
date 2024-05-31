# secrets
import os
from dotenv import load_dotenv
# db
import chromadb
from langchain_community.vectorstores import Chroma
# text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
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

#creates a persistent instance of Chroma that saves data to disk, useful for testing and development
chromadb_client = chromadb.PersistentClient(path="./Chroma_collections")
collection = chromadb_client.get_or_create_collection(name="cv-bot-db")

#Openai Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

langchain_chroma = Chroma(
    client=chromadb_client,
    collection_name="cv-bot-db",
    embedding_function=embeddings
)

# Set the relative path to the docs directory in the repository
current_dir = os.path.dirname(os.path.abspath(__file__))
doc_folder = os.path.join(current_dir, 'docs')

# Load data onto collection
# Create some metadata, chunk text and embed it

for counter, filename in enumerate(os.listdir(doc_folder), start=1):
    file_path = os.path.join(doc_folder, filename)
    loader = TextLoader(file_path, encoding='utf-8')
    data = loader.load()
    
    # generate document id
    document_id = f'id{counter}'
    #extract title
    document_title = filename.split('.')[0] 

    #splitter mode, chunk_size=maximum number of tokens, chunk_overlay between two chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, add_start_index=True)
    all_splits = text_splitter.split_documents(data)

    for doc_index, doc in enumerate(all_splits, start=1):
        # Aggiungi i metadati esistenti
        doc.metadata.update({'document_id': document_id, 'title': document_title})
        # Crea e aggiungi un chunk_id univoco 
        chunk_id = f'{document_id}_chunk{doc_index}'
        doc.metadata['chunk_id'] = chunk_id
    
    # Poi, passa all_splits direttamente a langchain_chroma
    langchain_chroma.add_documents(all_splits, embedding=embeddings)

query = st.chat_input(placeholder='Ask about Leonardo')
# query = "Leonardo's skills"
if query:
    embedding_vector = embeddings.embed_query(query) #--> in this case i have created a vector from the query


    docs = langchain_chroma.similarity_search_by_vector(embedding_vector, k=2)
    # print(f"Retrieved documents: {len(docs)}")

    template = """
                Given the following extracted parts of a long document and a question, create a final answer only with the context information.

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot:"""

    llm = ChatOpenAI(temperature=0,
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

    retriever = langchain_chroma.as_retriever(search_type='mmr')
    
    docs = retriever.invoke(query)
    output = chain.invoke({"input_documents": docs, "human_input": query}, return_only_outputs=True)

    with st.chat_message(name='user'):
        st.write(output['output_text'])