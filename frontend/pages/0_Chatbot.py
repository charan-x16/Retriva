"""Streamlit chatbot page for uploading PDFs and querying Retriva."""

import os
import re
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
SOURCE_TAG_RE = re.compile(r"\[Source:\s*page\s+(\d+),\s*[^\]]+\]")


def init_state():
    """Initialize Streamlit session state."""

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("indexed_docs", [])
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("show_chunks", False)


def ingest_pdf(uploaded_pdf):
    """Upload a PDF to the automatic backend ingestion endpoint."""

    files = {
        "file": (
            uploaded_pdf.name,
            uploaded_pdf.getvalue(),
            "application/pdf",
        )
    }
    response = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=1800)
    response.raise_for_status()
    return response.json()


def format_ingest_result(result):
    """Create a compact ingest success message."""

    messages = []
    text = result.get("text", {})
    visual = result.get("visual", {})
    if text.get("status") == "indexed":
        messages.append(f"{text['chunks']} OCR chunks")
    if visual.get("status") == "indexed":
        messages.append(f"{visual['pages']} visual pages")
    if messages:
        return "Indexed " + " and ".join(messages) + "."
    return "No retrievable content was indexed."


def ask_question(question):
    """Send a user question to the backend query endpoint."""

    response = requests.post(
        f"{BACKEND_URL}/query",
        json={"question": question},
        timeout=1800,
    )
    response.raise_for_status()
    return response.json()


def format_answer_for_chat(answer):
    """Show inline citations as compact page markers in the chat bubble."""

    return SOURCE_TAG_RE.sub(r"[p. \1]", answer or "")


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


def render_chunks(chunks, *, use_expander=True):
    """Render retrieved text chunks."""

    if not chunks:
        return

    if use_expander:
        with st.expander(f"Text evidence - {len(chunks)} chunks", expanded=False):
            _render_chunk_items(chunks)
        return

    st.markdown(f"**Text evidence - {len(chunks)} chunks**")
    _render_chunk_items(chunks)


def _render_chunk_items(chunks):
    """Render text chunk cards without wrapping layout."""

    for index, chunk in enumerate(chunks, start=1):
        page = chunk.get("page", "?")
        source = chunk.get("source", "unknown")
        rrf = chunk.get("rrf_score", 0)
        rerank_score = chunk.get("rerank_score", 0)

        with st.container(border=True):
            st.markdown(f"**#{index}** - Page {page} - `{source}`")
            st.caption(f"RRF {rrf:.3f} | Rerank {rerank_score:.3f}")
            st.write(chunk.get("text", ""))


def render_visual_results(results):
    """Render ColPali visual page retrieval results."""

    if not results:
        st.caption("No ColPali visual pages returned.")
        return

    for index, result in enumerate(results, start=1):
        page = result.get("page", "?")
        source = result.get("source", "unknown")
        score = result.get("score", 0)
        with st.container(border=True):
            st.markdown(f"**Page {page}**")
            st.caption(f"Rank {index} | Score {score:.3f}")
            st.write(source)


def render_evidence(message, *, use_expanders=True):
    """Render automatic text and visual evidence."""

    chunks = message.get("chunks", [])
    visual_results = message.get("visual_results", [])
    if chunks:
        render_chunks(chunks, use_expander=use_expanders)
    if visual_results:
        if not use_expanders:
            st.markdown(f"**Visual evidence - {len(visual_results)} pages**")
            render_visual_results(visual_results)
            return

        with st.expander(
            f"Visual evidence - {len(visual_results)} pages",
            expanded=False,
        ):
            render_visual_results(visual_results)


def render_correction_info(message):
    """Render CRAG correction metadata for an assistant response."""

    answer_mode = message.get("answer_mode")
    if answer_mode:
        labels = {
            "text_hybrid": "Text retrieval",
            "visual_multimodal": "Visual retrieval",
            "text_fallback": "Text fallback",
            "visual_missing_images": "Visual index needs re-ingestion",
            "generation_rate_limited": "Generation rate-limited",
            "generation_error": "Generation failed",
        }
        st.caption(f"Answer path: {labels.get(answer_mode, answer_mode)}")

    retrieval_summary = message.get("retrieval_summary") or {}
    if retrieval_summary:
        st.caption(
            "Retrieved "
            f"{retrieval_summary.get('text_chunks', 0)} text chunks and "
            f"{retrieval_summary.get('visual_pages', 0)} visual pages."
        )

    grade_score = message.get("grade_score")
    if grade_score is not None:
        st.caption(f"Context grade: {grade_score:.2f}")

    if message.get("was_corrected"):
        original_query = message.get("original_query", "")
        query_used = message.get("query_used", "")
        st.info(
            "Query was rewritten for better retrieval. "
            f"Original: {original_query} | Rewritten: {query_used}"
        )


def render_answer_details(message):
    """Render citations and diagnostics in one collapsed panel."""

    has_sources = bool(message.get("citations"))
    has_diagnostics = any(
        message.get(key) is not None
        for key in ("answer_mode", "retrieval_summary", "grade_score")
    ) or message.get("was_corrected")
    has_evidence = bool(message.get("chunks") or message.get("visual_results"))

    if not has_sources and not has_diagnostics and not has_evidence:
        return

    with st.expander("Answer details", expanded=False):
        render_sources(message.get("citations", []))
        render_correction_info(message)
        if st.session_state.show_chunks:
            render_evidence(message, use_expanders=False)


def render_message(message):
    """Render one chat message with optional citations and chunks."""

    with st.chat_message(message["role"]):
        content = message["content"]
        if message["role"] == "assistant":
            content = format_answer_for_chat(content)
        st.markdown(content)
        if message["role"] == "assistant":
            timestamp = message.get("timestamp")
            if timestamp:
                st.caption(timestamp)
            render_answer_details(message)


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
                    "visual_results": result.get("visual_results", []),
                    "answer_mode": result.get("answer_mode"),
                    "retrieval_summary": result.get("retrieval_summary"),
                    "was_corrected": result.get("was_corrected", False),
                    "grade_score": result.get("grade_score"),
                    "original_query": result.get("original_query", prompt),
                    "query_used": result.get("query_used", prompt),
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                }
                st.markdown(format_answer_for_chat(content))
                st.caption(assistant_message["timestamp"])
                render_answer_details(assistant_message)

    st.session_state.messages.append(assistant_message)


init_state()

with st.sidebar:
    st.title("Retriva")
    st.caption("Automatic retrieval-augmented chat over your PDFs")

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
                st.success(format_ingest_result(result))
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")

    if st.session_state.indexed_docs:
        st.divider()
        st.subheader("Library")
        for item in st.session_state.indexed_docs:
            source = item["source"]
            if len(source) > 30:
                source = source[:27] + "..."
            text_chunks = item.get("text", {}).get("chunks", 0)
            visual_pages = item.get("visual", {}).get("pages", 0)
            st.write(f"{source} - {text_chunks} chunks, {visual_pages} pages")

    st.divider()
    st.session_state.show_chunks = st.toggle(
        "Include retrieved evidence in details",
        value=st.session_state.show_chunks,
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

st.title("Chat with your documents")

if st.session_state.indexed_docs:
    doc_count = len(st.session_state.indexed_docs)
    chunk_count = sum(
        doc.get("text", {}).get("chunks", 0)
        for doc in st.session_state.indexed_docs
    )
    page_count = sum(
        doc.get("visual", {}).get("pages", 0)
        for doc in st.session_state.indexed_docs
    )
    st.caption(
        f"Ready - {doc_count} document(s), "
        f"{chunk_count} text chunks, {page_count} visual pages"
    )
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
