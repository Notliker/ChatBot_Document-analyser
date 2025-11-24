import streamlit as st

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Привет! Чем могу помочь?"}]
    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = []
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Привет! Чем могу помочь?"}]

