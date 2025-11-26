import logging
import os

import streamlit as st
from dotenv import load_dotenv

from rag import handle_user_query, MongoDB

load_dotenv()

LOG_LEVEL = os.getenv("CVBOT_LOG_LEVEL", "INFO").upper()
LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

app_logger = logging.getLogger("cv_bot.app")
if not app_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    app_logger.addHandler(handler)
app_logger.setLevel(LOG_LEVELS.get(LOG_LEVEL, logging.INFO))
app_logger.propagate = False

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
    app_logger.info("User query received: %s", prompt)
    display_chat_message("user", prompt)

    chat_history = build_chat_history()
    response, context = handle_user_query(
        query=prompt,
        collection=collection,
        chat_history=chat_history,
    )

    app_logger.info("Retrieved context chars: %d", len(context or ""))
    display_chat_message("assistant", response)

    with st.expander("Retrieved context", expanded=False):
        st.write(context or "No context retrieved for this query.")

st.button("Clear chat history", on_click=lambda: st.session_state.pop("messages", None))