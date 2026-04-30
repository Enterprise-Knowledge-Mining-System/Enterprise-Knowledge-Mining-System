from __future__ import annotations

import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from app.knowledge_pipeline import METADATA_KEYS, load_dotenv_if_available


INDEX_EXTENSIONS = (".parquet", ".jsonl", ".json", ".csv")
PDF_EXTENSIONS = (".pdf",)
CHROMA_ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz")
TEXT_COLUMNS = ("text", "document", "content", "chunk", "chunk_text")
EMBEDDING_COLUMNS = ("embedding", "embeddings", "vector", "vectors")
ID_COLUMNS = ("id", "ids", "chunk_id", "document_id")


def get_hf_credentials() -> tuple[str | None, str | None]:
    load_dotenv_if_available()
    return os.getenv("HF_REPO_ID"), os.getenv("HF_TOKEN")


def list_hf_files(repo_id: str, token: str | None = None) -> list[str]:
    return list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)


def list_hf_index_files(repo_id: str, token: str | None = None) -> list[str]:
    files = list_hf_files(repo_id, token)
    return sorted(file for file in files if file.lower().endswith(INDEX_EXTENSIONS))


def list_hf_pdf_files(repo_id: str, token: str | None = None) -> list[str]:
    files = list_hf_files(repo_id, token)
    return sorted(file for file in files if file.lower().endswith(PDF_EXTENSIONS))


def list_hf_chroma_archives(repo_id: str, token: str | None = None) -> list[str]:
    files = list_hf_files(repo_id, token)
    return sorted(file for file in files if file.lower().endswith(CHROMA_ARCHIVE_EXTENSIONS))


def download_hf_file(repo_id: str, filename: str, token: str | None = None) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
    )


def _is_chroma_root(path: Path) -> bool:
    return path.is_dir() and (path / "chroma.sqlite3").is_file()


def restore_chroma_archive(
    repo_id: str,
    filename: str,
    destination: str | Path,
    token: str | None = None,
) -> Path:
    archive_path = download_hf_file(repo_id, filename, token)
    destination = Path(destination)
    staging_dir = destination.with_name(f"{destination.name}_restore")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    lower_name = archive_path.name.lower()
    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(staging_dir)
    elif lower_name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive_path) as archive:
            archive.extractall(staging_dir)
    else:
        raise ValueError(f"Unsupported Chroma archive format: {archive_path.name}")

    candidates = [path for path in staging_dir.rglob("*") if _is_chroma_root(path)]
    if _is_chroma_root(staging_dir):
        source_dir = staging_dir
    elif candidates:
        source_dir = candidates[0]
    else:
        raise ValueError("Archive does not contain a Chroma database with chroma.sqlite3.")

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_dir, destination)
    shutil.rmtree(staging_dir, ignore_errors=True)
    return destination


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            for key in ("records", "chunks", "documents", "data"):
                if isinstance(data.get(key), list):
                    return data[key]
        if isinstance(data, list):
            return data
    raise ValueError(f"Unsupported Hugging Face index format: {path.name}")


def first_present(record: dict[str, Any], columns: tuple[str, ...]) -> str | None:
    return next((column for column in columns if column in record), None)


def parse_embedding(value: Any) -> list[float]:
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            value = json.loads(value)
        else:
            value = [part for part in value.split(",") if part.strip()]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("Embedding value must be a list, JSON list, or comma-separated string.")
    return [float(item) for item in value]


def parse_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    if isinstance(metadata, str) and metadata.strip():
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict):
        metadata = {}

    for key in METADATA_KEYS:
        if key in record and key not in metadata:
            metadata[key] = record[key]

    return {
        key: "" if value is None else value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def normalize_records(records: list[dict[str, Any]]) -> tuple[list[str], list[str], list[dict[str, Any]], list[list[float]]]:
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []

    for index, record in enumerate(records):
        text_column = first_present(record, TEXT_COLUMNS)
        embedding_column = first_present(record, EMBEDDING_COLUMNS)
        if not text_column or not embedding_column:
            continue

        document = record.get(text_column)
        if not isinstance(document, str) or not document.strip():
            continue

        id_column = first_present(record, ID_COLUMNS)
        chunk_id = str(record.get(id_column) or f"hf_chunk_{index}") if id_column else f"hf_chunk_{index}"

        ids.append(chunk_id)
        documents.append(document)
        metadatas.append(parse_metadata(record))
        embeddings.append(parse_embedding(record[embedding_column]))

    return ids, documents, metadatas, embeddings


def hydrate_collection_from_hf_index(
    collection: Any,
    repo_id: str,
    filename: str,
    token: str | None = None,
    batch_size: int = 1000,
) -> int:
    path = download_hf_file(repo_id, filename, token)
    ids, documents, metadatas, embeddings = normalize_records(load_records(path))
    for index in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[index : index + batch_size],
            documents=documents[index : index + batch_size],
            metadatas=metadatas[index : index + batch_size],
            embeddings=embeddings[index : index + batch_size],
        )
    return len(ids)
