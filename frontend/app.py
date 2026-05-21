"""Streamlit multipage router for Retriva."""

import streamlit as st

st.set_page_config(
    page_title="Retriva",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Retriva": [
        st.Page(
            "pages/0_Chatbot.py",
            title="Chatbot",
            icon=":material/chat:",
            default=True,
        ),
        st.Page(
            "pages/1_Evaluation.py",
            title="Evaluation",
            icon=":material/monitoring:",
        ),
    ]
}

navigation = st.navigation(pages, position="sidebar", expanded=True)
navigation.run()

