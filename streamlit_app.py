from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from hf_index_loader import (
    download_hf_file,
    get_hf_credentials,
    hydrate_collection_from_hf_index,
    list_hf_files,
    restore_chroma_archive,
)
from knowledge_pipeline import KnowledgeMiningPipeline, PipelineConfig, get_openai_api_key


st.set_page_config(
    page_title="Enterprise Knowledge Mining",
    layout="wide",
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


def chroma_db_exists(path: str) -> bool:
    return (Path(path) / "chroma.sqlite3").exists()


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
    st.caption("Uses HF_REPO_ID and HF_TOKEN from .env.")
    st.metric("Repository", "Configured" if repo_id else "Missing")

    index_files: list[str] = []
    pdf_files: list[str] = []
    chroma_archives: list[str] = []
    total_hf_files: int | None = None
    hf_error: str | None = None
    if repo_id:
        try:
            hf_files = cached_hf_files(repo_id, hf_token)
            total_hf_files = len(hf_files)
            index_files = [file for file in hf_files if file.lower().endswith((".parquet", ".jsonl", ".json", ".csv"))]
            pdf_files = [file for file in hf_files if file.lower().endswith(".pdf")]
            chroma_archives = [
                file
                for file in hf_files
                if file.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))
            ]
        except Exception as exc:
            hf_error = str(exc)

    if total_hf_files is not None:
        st.metric("HF files", f"{total_hf_files:,}")
    if pdf_files:
        st.metric("PDFs", f"{len(pdf_files):,}")

    ingest_count = st.number_input("PDFs to ingest", min_value=1, max_value=max(1, len(pdf_files)), value=min(25, max(1, len(pdf_files))))
    ingest_offset = st.number_input("Start offset", min_value=0, max_value=max(0, len(pdf_files) - 1), value=0)
    enrich_entities = st.toggle("Extract entities during ingestion", value=False)
    reset_before_ingest = st.toggle("Reset collection before ingest", value=False)
    ingest_pdfs = st.button("Ingest Hugging Face PDFs", disabled=not pdf_files or not get_openai_api_key())

    configured_archive = os.getenv("HF_CHROMA_ARCHIVE", "")
    archive_options = sorted(set(chroma_archives + ([configured_archive] if configured_archive else [])))
    selected_archive = st.selectbox("Chroma archive", archive_options) if archive_options else None
    restore_archive = st.button("Restore Chroma archive", disabled=not selected_archive)

    selected_index = st.selectbox("Chunk index file", index_files) if index_files else None
    sync_from_hf = st.button("Sync Hugging Face index", disabled=not selected_index)

if repo_id and selected_archive and (restore_archive or not chroma_db_exists(chroma_path)):
    with st.spinner("Restoring ChromaDB from Hugging Face..."):
        restore_chroma_archive(repo_id, selected_archive, chroma_path, hf_token)
    st.success("ChromaDB restored from Hugging Face.")

pipeline = get_pipeline(chroma_path, collection_name, embedding_model, rag_model)
collection = pipeline.get_collection()

if ingest_pdfs and repo_id:
    selected_pdfs = pdf_files[int(ingest_offset) : int(ingest_offset) + int(ingest_count)]
    progress = st.progress(0)
    status = st.empty()
    total_chunks = 0
    failed_files: list[str] = []

    for index, filename in enumerate(selected_pdfs, start=1):
        status.write(f"Ingesting {index:,}/{len(selected_pdfs):,}: {filename}")
        try:
            pdf_path = download_hf_file(repo_id, filename, hf_token)
            total_chunks += pipeline.ingest_pdf_file(
                pdf_path,
                enrich_entities=enrich_entities,
                reset_collection=reset_before_ingest and index == 1,
            )
        except Exception:
            failed_files.append(filename)
        progress.progress(index / len(selected_pdfs))

    if failed_files:
        st.warning(f"Ingested {total_chunks:,} chunks. {len(failed_files):,} PDFs failed.")
    else:
        st.success(f"Ingested {total_chunks:,} chunks from {len(selected_pdfs):,} PDFs.")

if sync_from_hf and repo_id and selected_index:
    with st.spinner("Downloading stored chunks and embeddings from Hugging Face..."):
        synced_count = hydrate_collection_from_hf_index(collection, repo_id, selected_index, hf_token)
    st.success(f"Synced {synced_count:,} embedded chunks from Hugging Face.")

chunk_count = collection.count()

status_col, key_col = st.columns([1, 1])
status_col.metric("Indexed chunks", f"{chunk_count:,}")
key_col.metric("OpenAI key", "Configured" if get_openai_api_key() else "Missing")

if hf_error:
    st.error(f"Could not read the Hugging Face dataset: {hf_error}")
elif repo_id and total_hf_files and not pdf_files and not index_files:
    st.warning("The Hugging Face dataset is reachable, but no PDFs or chunk index files were found.")

query = st.text_area(
    "Ask a question",
    placeholder="What methods do the indexed papers use for information retrieval?",
    height=110,
)

run_query = st.button("Ask", type="primary", disabled=not query.strip() or chunk_count == 0)

if chunk_count == 0:
    st.warning("No queryable chunks were found. Sync a Hugging Face chunk index or build the local Chroma collection first.")

if run_query:
    if not get_openai_api_key():
        st.error("Set OPENAI_KEY or OPENAI_API_KEY in .env before querying.")
        st.stop()

    with st.spinner("Retrieving context and generating an answer..."):
        response = pipeline.rag_query(
            query=query.strip(),
            n_results=n_results,
            min_similarity=min_similarity,
            filter_category=filter_category.strip() or None,
            use_hybrid=use_hybrid,
            entity_bonus=entity_bonus,
        )

    st.subheader("Answer")
    st.write(response["answer"])

    sources = response.get("sources", [])
    st.subheader("Sources")
    if sources:
        st.dataframe(source_table(sources), hide_index=True, use_container_width=True)
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
