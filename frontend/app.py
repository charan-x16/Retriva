"""Streamlit interface for uploading PDFs and querying Retriva."""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Retriva", page_icon="R", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f5ef;
        color: #1f2a2e;
    }
    section[data-testid="stSidebar"] {
        background: #e9e2d1;
        border-right: 1px solid #d8ccb4;
    }
    .stButton > button {
        border-radius: 6px;
        border: 1px solid #1f2a2e;
        background: #1f2a2e;
        color: #f7f5ef;
        font-weight: 700;
    }
    .stButton > button:disabled {
        border-color: #aaa292;
        background: #cfc5ae;
        color: #706957;
    }
    div[data-testid="stExpander"] {
        border-color: #d8ccb4;
        background: rgba(255, 255, 255, 0.45);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Retriva")
st.caption("Self-correcting hybrid document RAG")

with st.sidebar:
    st.header("Corpus")
    uploaded_pdf = st.file_uploader("PDF", type=["pdf"])
    ingest_clicked = st.button(
        "Ingest",
        type="primary",
        disabled=uploaded_pdf is None,
        use_container_width=True,
    )

    if ingest_clicked and uploaded_pdf is not None:
        files = {
            "file": (
                uploaded_pdf.name,
                uploaded_pdf.getvalue(),
                "application/pdf",
            )
        }
        with st.spinner("Indexing document"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ingest",
                    files=files,
                    timeout=900,
                )
                response.raise_for_status()
                result = response.json()
                st.success(
                    f"Indexed {result['chunks']} chunks from {result['source']}."
                )
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")

question = st.text_input("Question", placeholder="Ask about the indexed PDF")
submit_clicked = st.button(
    "Submit",
    type="primary",
    disabled=not question.strip(),
)

if submit_clicked:
    with st.spinner("Retrieving and answering"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"question": question},
                timeout=900,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Query failed: {exc}")
        else:
            st.subheader("Answer")
            st.write(result.get("answer", ""))

            st.subheader("Citations")
            citations = result.get("citations", [])
            if citations:
                for citation in citations:
                    st.write(
                        f"Page {citation.get('page')} - {citation.get('source')}"
                    )
            else:
                st.write("No inline citations returned.")

            with st.expander("Retrieved chunks"):
                for index, chunk in enumerate(result.get("chunks", []), start=1):
                    st.markdown(
                        f"**{index}. Page {chunk.get('page')} - "
                        f"{chunk.get('source')}**"
                    )
                    st.write(chunk.get("text", ""))
                    st.caption(
                        "RRF "
                        f"{chunk.get('rrf_score', 0):.4f} | "
                        "Rerank "
                        f"{chunk.get('rerank_score', 0):.4f}"
                    )

