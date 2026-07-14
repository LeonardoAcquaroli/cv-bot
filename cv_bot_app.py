import logging
import os

import streamlit as st
from dotenv import load_dotenv

from rag import get_qdrant_client, handle_user_query

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

# ---------------------------------------------------------------------------
# Page configuration (purely cosmetic, does not touch app logic)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chat with Leo",
    page_icon="👋🏻",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom styling (purely cosmetic, does not touch app logic)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Overall app background */
        .stApp {
            background: radial-gradient(circle at 10% 0%, #1f2a4d 0%, #0f1424 45%, #05070f 100%);
            color: #f2f2f7;
        }

        /* Hide default streamlit chrome for a cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Hero title */
        .hero-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 2.6rem;
            text-align: center;
            background: linear-gradient(90deg, #7dd3fc, #a78bfa 45%, #f472b6 90%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.2rem;
            padding-top: 0.5rem;
        }

        .hero-subtitle {
            text-align: center;
            color: #a3a9c2;
            font-size: 1.05rem;
            font-weight: 500;
            margin-bottom: 1.6rem;
        }

        .fancy-divider {
            height: 1px;
            border: none;
            background: linear-gradient(90deg, transparent, rgba(167,139,250,0.6), transparent);
            margin: 0.5rem 0 1.5rem 0;
        }

        /* Chat message bubbles */
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(6px);
        }

        [data-testid="stChatMessageAvatarUser"] {
            background: linear-gradient(135deg, #7dd3fc, #a78bfa) !important;
        }

        /* Chat input box - cosmetic only (border/radius/glow).
           Text colour, background and contrast are handled natively by
           .streamlit/config.toml so they can never end up mismatched. */
        [data-testid="stChatInput"] {
            border: 1px solid rgba(167, 139, 250, 0.35) !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: rgba(167, 139, 250, 0.7) !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #7dd3fc, #a78bfa 60%, #f472b6);
            color: #0f1424;
            font-weight: 600;
            border: none;
            border-radius: 999px;
            padding: 0.5rem 1.4rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            box-shadow: 0 4px 14px rgba(167, 139, 250, 0.35);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(167, 139, 250, 0.55);
            color: #0f1424;
        }

        /* Scrollbar polish */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(167, 139, 250, 0.4);
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-title">Chat with Leo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Ask me anything about Leonardo\'s CV, skills, and experience</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

avatar_image_url = 'https://raw.githubusercontent.com/LeonardoAcquaroli/cv-bot/main/avatar.webp'

qdrant_client = st.cache_resource(get_qdrant_client)()

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
        client=qdrant_client,
        chat_history=chat_history,
    )

    app_logger.info("Retrieved context chars: %d", len(context or ""))
    display_chat_message("assistant", response)

    # with st.expander("Retrieved context", expanded=False):
    #     st.write(context or "No context retrieved for this query.")

if st.session_state.get("messages"):
    st.button("Clear chat history", on_click=lambda: st.session_state.pop("messages", None))