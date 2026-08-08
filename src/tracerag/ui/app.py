from pathlib import Path

import streamlit as st

from tracerag.config import get_settings
from tracerag.ingestion.loaders import UnsupportedDocumentError
from tracerag.service import RAGService


@st.cache_resource
def get_service() -> RAGService:
    """Keep one in-process index for the lifetime of the Streamlit service."""
    return RAGService(get_settings())


service = get_service()

st.set_page_config(page_title="TraceRAG", page_icon="🔎", layout="wide")
st.title("TraceRAG")
st.caption("Evidence-first answers from your documents")

with st.sidebar:
    st.header("Knowledge base")
    st.caption(f"Retrieval mode: `{service.retrieval_mode}`")
    st.caption(f"Indexed chunks: {service.chunk_count}")
    uploads = st.file_uploader(
        "Upload PDF, Markdown, or text files",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Build index", type="primary", disabled=not uploads):
        upload_dir = service.settings.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        try:
            with st.spinner("Parsing and indexing documents..."):
                for upload in uploads:
                    path = upload_dir / Path(upload.name).name
                    path.write_bytes(upload.getvalue())
                    paths.append(path)
                chunks = service.ingest(paths)
            st.success(f"Indexed {chunks} chunks from {len(paths)} files.")
        except (UnsupportedDocumentError, OSError, ValueError) as exc:
            st.error(f"Indexing failed: {exc}")

question = st.chat_input("Ask a question about the indexed documents")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence..."):
            result = service.ask(question)
        st.write(result.text)
        st.caption(f"Confidence: {result.confidence:.0%}")
        for citation in result.citations:
            with st.expander(f"[{citation.source_id}] {citation.filename} · page {citation.page}"):
                st.write(citation.excerpt)
