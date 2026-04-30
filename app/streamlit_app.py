from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from app.hf_index_loader import (
    get_hf_credentials,
    list_hf_files,
    restore_chroma_archive,
)
from app.knowledge_pipeline import KnowledgeMiningPipeline, PipelineConfig, get_openai_api_key


DEFAULT_CHROMA_ARCHIVE = "chroma_db.zip"


st.set_page_config(
    page_title="Enterprise Knowledge Mining",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_streamlit_secrets() -> None:
    for key in ("OPENAI_KEY", "OPENAI_API_KEY", "HF_REPO_ID", "HF_TOKEN", "HF_CHROMA_ARCHIVE"):
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value and not os.getenv(key):
            os.environ[key] = str(value)


@st.cache_resource(show_spinner=False)
def get_pipeline(
    chroma_path: str,
    collection_name: str,
    embedding_model: str,
    rag_model: str,
) -> KnowledgeMiningPipeline:
    config = PipelineConfig(
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
        rag_model=rag_model,
    )
    return KnowledgeMiningPipeline(config=config)


@st.cache_resource(show_spinner=False)
def get_chroma_collection(chroma_path: str, collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
    )


def format_score(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def source_table(sources: list[dict]) -> pd.DataFrame:
    rows = []
    for source in sources:
        rows.append(
            {
                "#": source.get("source_number"),
                "Title": source.get("title"),
                "Section": source.get("section"),
                "Page": source.get("page_number"),
                "Category": source.get("primary_category"),
                "Similarity": format_score(source.get("cosine_similarity")),
                "Hybrid": format_score(source.get("hybrid_score")),
                "Chunk": source.get("chunk_index"),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def cached_hf_files(repo_id: str, token: str | None) -> list[str]:
    return list_hf_files(repo_id, token)


def chroma_collection_has_chunks(path: str, collection_name: str) -> bool:
    if not (Path(path) / "chroma.sqlite3").exists():
        return False

    try:
        import chromadb

        client = chromadb.PersistentClient(path=path)
        collection = client.get_collection(collection_name)
        return collection.count() > 0
    except Exception:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def count_indexed_papers(path: str, collection_name: str, batch_size: int = 5000) -> int:
    if not (Path(path) / "chroma.sqlite3").exists():
        return 0

    try:
        import chromadb

        client = chromadb.PersistentClient(path=path)
        collection = client.get_collection(collection_name)
        total_chunks = collection.count()
        paper_ids: set[str] = set()

        for offset in range(0, total_chunks, batch_size):
            batch = collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            for metadata in batch.get("metadatas") or []:
                if not metadata:
                    continue
                paper_id = metadata.get("paper_id") or metadata.get("title")
                if paper_id:
                    paper_ids.add(str(paper_id))

        return len(paper_ids)
    except Exception:
        return 0


load_streamlit_secrets()

st.title("Enterprise Knowledge Mining")

with st.sidebar:
    repo_id, hf_token = get_hf_credentials()
    st.header("Query Settings")
    chroma_path = st.text_input("Chroma path", value="./chroma_db")
    collection_name = st.text_input("Collection", value="research_papers")
    embedding_model = st.text_input("Embedding model", value="text-embedding-3-small")
    rag_model = st.text_input("RAG model", value="gpt-4.1-mini")
    n_results = st.slider("Sources", min_value=1, max_value=12, value=5)
    min_similarity = st.slider("Minimum similarity", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    filter_category = st.text_input("Category filter", placeholder="Example: cs.AI")
    use_hybrid = st.toggle("Hybrid entity reranking", value=False)
    entity_bonus = st.slider("Entity bonus", min_value=0.0, max_value=2.0, value=0.5, step=0.1, disabled=not use_hybrid)

    st.header("Hugging Face Corpus")
    st.caption("Uses HF_REPO_ID and HF_TOKEN from Streamlit secrets or .env.")
    st.metric("Repository", "Configured" if repo_id else "Missing")
    refresh_hf_files = st.button("Load Hugging Face files", disabled=not repo_id)

    chroma_archives: list[str] = []
    total_hf_files: int | None = None
    hf_error: str | None = None
    if repo_id and refresh_hf_files:
        try:
            hf_files = cached_hf_files(repo_id, hf_token)
            st.session_state.hf_files = hf_files
        except Exception as exc:
            hf_error = str(exc)

    hf_files = st.session_state.get("hf_files", [])
    if repo_id and hf_files:
        try:
            total_hf_files = len(hf_files)
            chroma_archives = [
                file
                for file in hf_files
                if file.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))
            ]
        except Exception as exc:
            hf_error = str(exc)

    if total_hf_files is not None:
        st.metric("HF files", f"{total_hf_files:,}")

    configured_archive = os.getenv("HF_CHROMA_ARCHIVE", DEFAULT_CHROMA_ARCHIVE)
    archive_options = sorted(set(chroma_archives + [configured_archive]))
    archive_index = archive_options.index(configured_archive) if configured_archive in archive_options else 0
    selected_archive = st.selectbox("Chroma archive", archive_options, index=archive_index) if archive_options else None
    restore_archive = st.button("Restore Chroma archive", disabled=not selected_archive)

has_queryable_chunks = chroma_collection_has_chunks(chroma_path, collection_name)

if repo_id and selected_archive and restore_archive:
    with st.spinner("Restoring ChromaDB from Hugging Face..."):
        try:
            restore_chroma_archive(repo_id, selected_archive, chroma_path, hf_token)
        except Exception as exc:
            st.error(f"Could not restore ChromaDB archive '{selected_archive}' from Hugging Face: {exc}")
            st.stop()
    get_pipeline.clear()
    has_queryable_chunks = chroma_collection_has_chunks(chroma_path, collection_name)
    if not has_queryable_chunks:
        st.error(
            f"Restored '{selected_archive}', but collection '{collection_name}' still has no chunks. "
            "Check that the archive contains the same Chroma collection name."
        )
        st.stop()
    st.success("ChromaDB restored from Hugging Face.")

try:
    collection = get_chroma_collection(chroma_path, collection_name)
except Exception as exc:
    st.error(f"Could not initialize ChromaDB at '{chroma_path}': {exc}")
    st.stop()

chunk_count = collection.count()
paper_count = count_indexed_papers(chroma_path, collection_name)
last_response_time = st.session_state.get("last_query_response_time")

papers_col, chunks_col, response_time_col = st.columns(3)
papers_col.metric("Indexed papers", f"{paper_count:,}")
chunks_col.metric("Indexed chunks", f"{chunk_count:,}")
response_time_slot = response_time_col.empty()
response_time_slot.metric(
    "Query response time",
    f"{last_response_time:.2f}s" if last_response_time is not None else "Not run",
)

if chunk_count == 0:
    st.info(
        "No indexed data is loaded in this deployment yet. "
        "Streamlit Cloud does not include local ignored folders like chroma_db/, "
        "so restore the Chroma archive from Hugging Face."
    )

    setup_col, action_col = st.columns([2, 1])
    with setup_col:
        st.write(
            {
                "HF_REPO_ID": "Configured" if repo_id else "Missing",
                "OPENAI API key": "Configured" if get_openai_api_key() else "Missing",
                "Chroma archive": selected_archive or "Not found",
            }
        )
    with action_col:
        if selected_archive:
            st.caption("Use the sidebar Restore button to load the Chroma archive.")
        elif not repo_id:
            st.caption("Add HF_REPO_ID in Streamlit app secrets.")
        else:
            st.caption("No Chroma archive was found in the Hugging Face dataset.")

if hf_error:
    st.error(f"Could not read the Hugging Face dataset: {hf_error}")
elif repo_id and total_hf_files and not chroma_archives:
    st.warning("The Hugging Face dataset is reachable, but no Chroma archive files were found.")

query = st.text_area(
    "Ask a question",
    placeholder="What methods do the indexed papers use for information retrieval?",
    height=110,
)

run_query = st.button("Ask", type="primary", disabled=chunk_count == 0)

if run_query:
    query_text = query.strip()
    if not query_text:
        st.warning("Enter a question before asking.")
        st.stop()

    if not get_openai_api_key():
        st.error("Set OPENAI_KEY or OPENAI_API_KEY in Streamlit secrets before querying.")
        st.stop()

    pipeline = get_pipeline(chroma_path, collection_name, embedding_model, rag_model)
    with st.spinner("Retrieving context and generating an answer..."):
        query_start_time = time.perf_counter()
        response = pipeline.rag_query(
            query=query_text,
            n_results=n_results,
            min_similarity=min_similarity,
            filter_category=filter_category.strip() or None,
            use_hybrid=use_hybrid,
            entity_bonus=entity_bonus,
        )
        st.session_state.last_query_response_time = time.perf_counter() - query_start_time
        response_time_slot.metric("Query response time", f"{st.session_state.last_query_response_time:.2f}s")

    st.subheader("Answer")
    st.write(response["answer"])

    sources = response.get("sources", [])
    st.subheader("Sources")
    if sources:
        st.dataframe(source_table(sources), hide_index=True, width="stretch")
    else:
        st.info("No sources met the retrieval threshold.")

    with st.expander("Retrieved chunks"):
        for source in sources:
            chunk_id = source.get("chunk_id", "")
            st.markdown(
                f"**Source {source.get('source_number')}**  \n"
                f"{source.get('title', '')}  \n"
                f"Section: {source.get('section', '')} | Page: {source.get('page_number', '')} | Chunk: {chunk_id}"
            )
            st.write(source.get("text", ""))
