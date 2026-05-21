"""Streamlit chatbot interface for uploading PDFs and querying Retriva."""

import os
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Retriva",
    page_icon="R",
    layout="centered",
    initial_sidebar_state="expanded",
)


def init_state():
    """Initialize Streamlit session state."""

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("indexed_docs", [])
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("show_chunks", False)


def ingest_pdf(uploaded_pdf):
    """Upload a PDF to the backend ingestion endpoint."""

    files = {
        "file": (
            uploaded_pdf.name,
            uploaded_pdf.getvalue(),
            "application/pdf",
        )
    }
    response = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=900)
    response.raise_for_status()
    return response.json()


def ask_question(question):
    """Send a user question to the backend query endpoint."""

    response = requests.post(
        f"{BACKEND_URL}/query",
        json={"question": question},
        timeout=900,
    )
    response.raise_for_status()
    return response.json()


def render_sources(citations):
    """Render citations in the same format expected from the answer."""

    if not citations:
        return

    lines = ["**Sources**"]
    for citation in citations:
        page = citation.get("page", "?")
        source = str(citation.get("source", "unknown"))
        lines.append(f"- Source: page {page}, {source}")
    st.markdown("\n".join(lines))


def render_chunks(chunks):
    """Render retrieved chunks inside an expander."""

    if not chunks:
        return

    with st.expander(f"Retrieved evidence - {len(chunks)} chunks", expanded=False):
        for index, chunk in enumerate(chunks, start=1):
            page = chunk.get("page", "?")
            source = chunk.get("source", "unknown")
            rrf = chunk.get("rrf_score", 0)
            rerank_score = chunk.get("rerank_score", 0)

            with st.container(border=True):
                st.markdown(f"**#{index}** - Page {page} - `{source}`")
                st.caption(f"RRF {rrf:.3f} | Rerank {rerank_score:.3f}")
                st.write(chunk.get("text", ""))


def render_message(message):
    """Render one chat message with optional citations and chunks."""

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            timestamp = message.get("timestamp")
            if timestamp:
                st.caption(timestamp)
            render_sources(message.get("citations", []))
            if st.session_state.show_chunks:
                render_chunks(message.get("chunks", []))


def handle_query(prompt):
    """Send a query and append the assistant response to chat history."""

    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%I:%M %p"),
    }
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence"):
            try:
                result = ask_question(prompt)
            except requests.RequestException as exc:
                content = f"Query failed: {exc}"
                st.error(content)
                assistant_message = {
                    "role": "assistant",
                    "content": content,
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                }
            else:
                content = result.get("answer") or "_No answer returned._"
                assistant_message = {
                    "role": "assistant",
                    "content": content,
                    "citations": result.get("citations", []),
                    "chunks": result.get("chunks", []),
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                }
                st.markdown(content)
                st.caption(assistant_message["timestamp"])
                render_sources(assistant_message["citations"])
                if st.session_state.show_chunks:
                    render_chunks(assistant_message["chunks"])

    st.session_state.messages.append(assistant_message)


init_state()

with st.sidebar:
    st.title("Retriva")
    st.caption("Retrieval-augmented chat over your PDFs")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_pdf is not None:
        size_kb = len(uploaded_pdf.getvalue()) / 1024
        st.caption(f"{uploaded_pdf.name} - {size_kb:,.1f} KB")

    ingest_clicked = st.button(
        "Ingest document",
        type="primary",
        disabled=uploaded_pdf is None,
        use_container_width=True,
    )

    if ingest_clicked and uploaded_pdf is not None:
        with st.spinner(f"Indexing {uploaded_pdf.name}"):
            try:
                result = ingest_pdf(uploaded_pdf)
                st.session_state.indexed_docs.append(result)
                st.success(
                    f"Indexed {result['chunks']} chunks from {result['source']}."
                )
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")

    if st.session_state.indexed_docs:
        st.divider()
        st.subheader("Library")
        for item in st.session_state.indexed_docs:
            source = item["source"]
            if len(source) > 30:
                source = source[:27] + "..."
            st.write(f"{source} - {item['chunks']} chunks")

    st.divider()
    st.session_state.show_chunks = st.toggle(
        "Show retrieved chunks",
        value=st.session_state.show_chunks,
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

st.title("Chat with your documents")

if st.session_state.indexed_docs:
    doc_count = len(st.session_state.indexed_docs)
    chunk_count = sum(doc.get("chunks", 0) for doc in st.session_state.indexed_docs)
    st.caption(f"Ready - {doc_count} document(s), {chunk_count} chunks")
else:
    st.caption("Upload and ingest a PDF, then ask a question.")

if not st.session_state.messages and st.session_state.indexed_docs:
    st.markdown("##### Try a quick prompt")
    suggestions = [
        ("Summarize", "Give me a concise summary of this document."),
        ("Key findings", "What are the key findings and takeaways?"),
        ("Methodology", "Explain the methodology used in detail."),
        ("Conclusions", "What conclusions does the author draw?"),
    ]
    cols = st.columns(2)
    for index, (label, full_prompt) in enumerate(suggestions):
        with cols[index % 2]:
            if st.button(label, key=f"suggest_{index}", use_container_width=True):
                st.session_state.pending_prompt = full_prompt

for chat_message in st.session_state.messages:
    render_message(chat_message)

if st.session_state.pending_prompt:
    pending_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    handle_query(pending_prompt)

prompt = st.chat_input("Ask about your document")
if prompt:
    handle_query(prompt)
