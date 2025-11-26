import os
from dotenv import load_dotenv
import streamlit as st

from rag import handle_user_query, MongoDB

load_dotenv()

st.title("Chat with Leo👋🏻")

avatar_image_url = 'https://raw.githubusercontent.com/LeonardoAcquaroli/cv-bot/main/avatar.webp'

pymongo = MongoDB()
collection = pymongo.get_collection('my_documents')

st.markdown("### Try the chat!🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar_image_url if message["role"] == "assistant" else None):
        st.markdown(message["content"])


def display_chat_message(role, content):
    """Render a chat bubble and persist it to session state."""
    with st.chat_message(role, avatar=avatar_image_url if role == "assistant" else None):
        st.markdown(content)
    st.session_state.messages.append({"role": role, "content": content})


def build_chat_history():
    return [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]


if prompt := st.chat_input(placeholder='''For example: "What are Leonardo's soft skills?"'''):
    display_chat_message("user", prompt)

    chat_history = build_chat_history()
    response, _ = handle_user_query(
        query=prompt,
        collection=collection,
        chat_history=chat_history,
    )

    display_chat_message("assistant", response)

st.button("Clear chat history", on_click=lambda: st.session_state.pop("messages", None))