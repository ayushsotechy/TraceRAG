import os

import httpx
import streamlit as st

API_URL = os.getenv("TRACERAG_API_URL", "http://localhost:8000").rstrip("/")
if not API_URL.startswith(("http://", "https://")):
    API_URL = f"http://{API_URL}"

st.set_page_config(page_title="TraceRAG", page_icon="🔎", layout="wide")
st.title("TraceRAG")
st.caption("Evidence-first answers from your documents")

with st.sidebar:
    st.header("Knowledge base")
    try:
        response = httpx.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        health = response.json()
        st.caption(f"Retrieval mode: `{health['retrieval_mode']}`")
        st.caption(f"Indexed chunks: {health['chunks']}")
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        st.warning("The API is waking up. Please refresh in a few seconds.")
    uploads = st.file_uploader(
        "Upload PDF, Markdown, or text files",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Build index", type="primary", disabled=not uploads):
        files = [("files", (item.name, item.getvalue(), item.type)) for item in uploads]
        try:
            with st.spinner("Parsing and indexing documents..."):
                response = httpx.post(f"{API_URL}/v1/documents", files=files, timeout=120)
                response.raise_for_status()
                payload = response.json()
            st.success(f"Indexed {payload['chunks']} chunks from {payload['files']} files.")
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            st.error("Indexing failed while the API was unavailable. Please try again.")

question = st.chat_input("Ask a question about the indexed documents")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving evidence..."):
                response = httpx.post(
                    f"{API_URL}/v1/query",
                    json={"question": question},
                    timeout=60,
                )
                response.raise_for_status()
                payload = response.json()
            st.write(payload["answer"])
            st.caption(f"Confidence: {payload['confidence']:.0%}")
            for citation in payload["citations"]:
                with st.expander(
                    f"[{citation['source_id']}] {citation['filename']} · page {citation['page']}"
                ):
                    st.write(citation["excerpt"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            st.error("The API could not answer right now. Please retry in a few seconds.")
