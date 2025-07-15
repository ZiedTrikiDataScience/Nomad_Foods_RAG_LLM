# streamlit_agent_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from rag_nomad_foods_chatbot_langchain import run_agent
from PIL import Image

# 0. Page config: MUST be first Streamlit call
st.set_page_config(page_title="Nomad Foods Agent", page_icon="🤖", layout="wide")

load_dotenv()

def load_logo():
    try:
        return Image.open("nomad_foods_logo.jpg")
    except FileNotFoundError:
        return None

def app():
    st.title("🤖 Nomad Foods AI Agent")

    logo = load_logo()
    if logo:
        st.image(logo, width=200)

    prompt = st.text_input("Ask me anything about Nomad Foods:")
    if prompt:
        with st.spinner("Thinking..."):
            answer = run_agent(prompt)
        st.markdown(f"**Agent:** {answer}")

if __name__ == "__main__":
    app()
