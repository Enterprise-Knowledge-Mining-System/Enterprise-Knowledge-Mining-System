from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_URL = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
EXCLUDED_SECTIONS = {
    "References",
    "Acknowledgment",
    "Acknowledgements",
}
METADATA_KEYS = [
    "paper_id",
    "title",
    "section",
    "primary_category",
    "categories",
    "authors",
    "published",
    "creation_date",
    "page_number",
    "page_count",
    "format",
    "page_block_index",
    "page_chunk_index",
    "chunk_index",
    "entities",
    "entity_labels",
]
RAG_SYSTEM_PROMPT = """
You are an academic research assistant for an enterprise knowledge mining system built on arXiv papers.

Instructions:
- Answer questions only from the retrieved context.
- Treat the retrieved text as excerpts from research papers.
- Prefer precise academic wording.
- Summarize contributions, methods, datasets, results, or limitations only if they are supported by the context.
- If the answer is incomplete or missing from the context, say that explicitly.
- Do not include source tags or citation brackets such as [Source 1] in the answer text.
- Write only the answer and ensure it is formatted concisely and clearly.
"""


def load_dotenv_if_available() -> None:
    try:
        import dotenv
    except ModuleNotFoundError:
        return
    dotenv.load_dotenv()


def get_openai_api_key() -> str | None:
    return os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")


@dataclass
class PipelineConfig:
    member_mode: bool = True
    search_query: str = "all:computer"
    start: int = 0
    max_results: int = 1000
    pdf_dir: Path = Path("pdfs")
    min_page_length: int = 100
    chunk_size: int = 800
    chunk_overlap: int = 50
    batch_size: int = 200
    embedding_model: str = "text-embedding-3-small"
    rag_model: str = "gpt-4.1-mini"
    chroma_path: str = "./chroma_db"
    collection_name: str = "research_papers"
    download_workers: int = min(12, (os.cpu_count() or 4) * 2)
    process_workers: int = max(1, (os.cpu_count() or 4) - 1)
    arxiv_page_size: int = 100
    request_delay: float = 3.0
    max_embed_workers: int = 3
    tpm_limit: int = 900_000

    @property
    def arxiv_params(self) -> dict[str, Any]:
        return {
            "search_query": self.search_query,
            "start": self.start,
            "max_results": self.max_results,
        }


class HuggingFacePdfCache:
    def __init__(self, repo_id: str | None = None, token: str | None = None):
        from huggingface_hub import HfApi

        load_dotenv_if_available()
        self.repo_id = repo_id or os.getenv("HF_REPO_ID")
        self.token = token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.token) if self.token else HfApi()
        self._files_cache: set[str] | None = None
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        return bool(self.repo_id)

    def get_files(self) -> set[str]:
        if not self.is_configured:
            return set()
        with self._lock:
            if self._files_cache is None:
                try:
                    from huggingface_hub import list_repo_files

                    self._files_cache = set(
                        list_repo_files(
                            repo_id=self.repo_id,
                            repo_type="dataset",
                            token=self.token,
                        )
                    )
                except Exception:
                    self._files_cache = set()
            return self._files_cache

    def exists(self, filename: str) -> bool:
        return filename in self.get_files()

    def upload(self, local_path: str | Path) -> None:
        from huggingface_hub.utils import disable_progress_bars

        if not self.is_configured:
            return
        filename = Path(local_path).name
        disable_progress_bars()
        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.token,
        )
        with self._lock:
            if self._files_cache is not None:
                self._files_cache.add(filename)

    def download(self, filename: str, local_path: str | Path) -> None:
        from huggingface_hub import hf_hub_download

        if not self.is_configured:
            raise RuntimeError("HF_REPO_ID is not configured.")
        hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(Path(local_path).parent),
            token=self.token,
        )


def make_request(url: str, delay: float = 5, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "arxiv-data-project/1.0 (research script)"},
            )
            with urllib.request.urlopen(request) as response:
                data = response.read()
            time.sleep(delay)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            time.sleep(delay * (2**attempt))
    raise RuntimeError("Max retries exceeded")


def get_text(entry: ET.Element, tag: str) -> str:
    return entry.findtext(tag, default="", namespaces=NS).strip()


def extract_authors(entry: ET.Element) -> list[str]:
    return [
        author.find("atom:name", NS).text.strip()
        for author in entry.findall("atom:author", NS)
        if author.find("atom:name", NS) is not None
    ]


def extract_categories(entry: ET.Element) -> list[str]:
    return [
        cat.attrib.get("term")
        for cat in entry.findall("atom:category", NS)
        if cat.attrib.get("term")
    ]


def extract_pdf_link(entry: ET.Element) -> str | None:
    return next(
        (
            link.attrib.get("href")
            for link in entry.findall("atom:link", NS)
            if link.attrib.get("title") == "pdf"
            and link.attrib.get("type") == "application/pdf"
            and link.attrib.get("rel") == "related"
        ),
        None,
    )


def extract_primary_category(entry: ET.Element) -> str | None:
    primary = entry.find("arxiv:primary_category", NS)
    return primary.attrib.get("term") if primary is not None else None


def extract_paper(entry: ET.Element) -> dict[str, Any]:
    return {
        "arxiv_id": get_text(entry, "atom:id").split("/")[-1],
        "title": get_text(entry, "atom:title"),
        "summary": get_text(entry, "atom:summary"),
        "published": get_text(entry, "atom:published"),
        "authors": extract_authors(entry),
        "primary_category": extract_primary_category(entry),
        "categories": extract_categories(entry),
        "pdf_link": extract_pdf_link(entry),
    }


def parse_response(response: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(response)
    return [extract_paper(entry) for entry in root.findall("atom:entry", NS)]


def fetch_all_papers(
    params: dict[str, Any],
    base_url: str = BASE_URL,
    page_size: int = 100,
    delay: float = 3.0,
) -> list[dict[str, Any]]:
    total_needed = params.get("max_results", 10)
    all_papers = []
    fetched = 0

    while fetched < total_needed:
        batch = min(page_size, total_needed - fetched)
        page_params = {
            **params,
            "start": params.get("start", 0) + fetched,
            "max_results": batch,
        }
        query_url = base_url + "?" + urllib.parse.urlencode(page_params)
        response = make_request(query_url, delay=delay)
        page_papers = parse_response(response)
        if not page_papers:
            break
        all_papers.extend(page_papers)
        fetched += len(page_papers)

    return all_papers


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def paper_from_pdf_path(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    title = path.stem.replace("_", " ").strip() or "untitled"
    paper_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "paper_0"
    return {
        "arxiv_id": paper_id,
        "title": title,
        "summary": "",
        "published": "",
        "authors": [],
        "primary_category": "",
        "categories": [],
        "pdf_link": "",
    }


def download_pdf(url: str, save_path: str | Path, overwrite: bool = False, delay: float = 1.0) -> Path:
    save_path = Path(save_path)
    if save_path.exists() and not overwrite:
        return save_path
    data = make_request(url, delay=delay, retries=3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(data)
    return save_path


def parse_pdf(paper: dict[str, Any], path: str | Path) -> dict[str, Any]:
    import pymupdf as fitz

    doc = fitz.open(path)
    pages = []
    try:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            pages.append(
                {
                    "text": page_text,
                    "raw_text": page_text,
                    "metadata": {
                        "page_number": page_num,
                        "page_count": len(doc),
                        "format": doc.metadata.get("format", ""),
                        "creationDate": doc.metadata.get("creationDate", ""),
                    },
                }
            )
        return {**paper, "pages": pages}
    finally:
        doc.close()


def download_and_parse_papers(
    papers: list[dict[str, Any]],
    config: PipelineConfig,
    hf_cache: HuggingFacePdfCache | None = None,
) -> list[dict[str, Any]]:
    hf_cache = hf_cache or HuggingFacePdfCache()
    config.pdf_dir.mkdir(parents=True, exist_ok=True)
    arxiv_semaphore = threading.Semaphore(4)

    def download_one(paper: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
        if not paper.get("pdf_link"):
            return paper, None

        filename = f"{sanitize_filename(paper['title'])}.pdf"
        local_path = config.pdf_dir / filename

        if local_path.exists():
            return paper, local_path

        if hf_cache.exists(filename):
            try:
                hf_cache.download(filename, local_path)
                return paper, local_path
            except Exception:
                if config.member_mode:
                    return paper, None

        if config.member_mode:
            return paper, None

        try:
            with arxiv_semaphore:
                download_pdf(paper["pdf_link"], local_path, delay=1.0)
            hf_cache.upload(local_path)
            return paper, local_path
        except Exception:
            return paper, None

    downloadable = [paper for paper in papers if paper.get("pdf_link")]
    downloaded = []
    with ThreadPoolExecutor(max_workers=config.download_workers) as executor:
        futures = {executor.submit(download_one, paper): paper for paper in downloadable}
        for future in as_completed(futures):
            paper, path = future.result()
            if path:
                downloaded.append((paper, path))

    processed_papers = []
    with ThreadPoolExecutor(max_workers=config.process_workers) as executor:
        futures = {executor.submit(parse_pdf, paper, path): (paper, path) for paper, path in downloaded}
        for future in as_completed(futures):
            try:
                processed_papers.append(future.result())
            except Exception:
                continue

    return processed_papers


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^\s*#{1,6}\s*", " ", text)
    text = re.sub(r"(?m)^\s*[-*_~=`]{2,}\s*$", " ", text)
    text = re.sub(r"[_*`~]+", " ", text)
    text = re.sub(r"(?<!\w)#(?=\w)", "", text)
    text = text.replace("#", " ")
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)
    text = re.sub(r"(?im)^\s*page\s*\d+(\s*of\s*\d+)?\s*$", " ", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", " ", text)
    text = re.sub(r"[-]+", " - ", text)
    text = re.sub(r"^[\s\.,;:-]+", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_page_text(text: str, page_number: int | None = None) -> str:
    text = clean_text(text)
    if not text:
        return ""
    if page_number == 1:
        match = re.search(r"\babstract\b\s*[:-]?\s*", text, flags=re.IGNORECASE)
        if match:
            text = text[match.start() :]
    return re.sub(r"\s+", " ", text).strip()


def clean_processed_papers(processed_papers: list[dict[str, Any]], min_length: int) -> list[dict[str, Any]]:
    cleaned_papers = []
    for paper in processed_papers:
        clean_pages = []
        for page in paper["pages"]:
            raw_text = page.get("raw_text", "") or page.get("text", "")
            cleaned = clean_page_text(raw_text, page_number=page["metadata"].get("page_number"))
            if cleaned and len(cleaned) >= min_length:
                clean_pages.append({"text": cleaned, "raw_text": raw_text, "metadata": page["metadata"]})
        cleaned_papers.append({**{key: value for key, value in paper.items() if key != "pages"}, "pages": clean_pages})
    return cleaned_papers


def extract_section_blocks(page_text: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"^(##.*|_?\*{0,2}Abstract\*{0,2}_?.*|_?\*{0,2}Keywords:.*\*{0,2}_?)$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(page_text))
    section_blocks = []
    for index, match in enumerate(matches):
        section = re.sub(r"[*_`#]+", "", match.group()).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(page_text)
        text = page_text[start:end].strip()
        if text:
            section_blocks.append({"section": section, "text": text})
    return section_blocks


def build_chunk_paper_metadata(paper: dict[str, Any]) -> dict[str, Any]:
    categories = paper.get("categories", [])
    return {
        "paper_id": paper.get("arxiv_id", "paper_0"),
        "title": paper.get("title", "untitled"),
        "primary_category": paper.get("primary_category", ""),
        "categories": ", ".join(categories) if isinstance(categories, list) else str(categories),
        "authors": ", ".join(paper.get("authors", [])) if paper.get("authors") else "",
        "published": paper.get("published", ""),
    }


def build_chunk_page_metadata(page: dict[str, Any]) -> dict[str, Any]:
    metadata = page.get("metadata", {})
    return {
        "creation_date": metadata.get("creationDate", ""),
        "page_number": metadata.get("page_number", ""),
        "page_count": metadata.get("page_count", ""),
        "format": metadata.get("format", ""),
    }


def remove_known_running_headers(text: str) -> str:
    return re.sub(
        r"(?i)\(?IJCSIS\)?\s+International Journal of Computer Science and Information Security,\s*Vol\.\s*7,\s*No\.\s*1,\s*2010",
        " ",
        text or "",
    )


def build_section_block(
    section: str,
    text: str,
    paper: dict[str, Any],
    page: dict[str, Any],
    page_block_index: int,
) -> dict[str, Any] | None:
    block_text = clean_text(remove_known_running_headers(text))
    if not block_text:
        return None
    return {
        "section": section,
        "text": block_text,
        "metadata": {
            **build_chunk_paper_metadata(paper),
            **build_chunk_page_metadata(page),
            "page_block_index": page_block_index,
        },
    }


def extract_blocks_from_page(
    page: dict[str, Any],
    paper: dict[str, Any] | None = None,
    excluded_sections: set[str] | None = None,
) -> list[dict[str, Any]]:
    paper = paper or {}
    excluded_sections = excluded_sections or set()
    extracted_blocks = extract_section_blocks(page.get("raw_text") or page.get("text", ""))
    blocks = []
    for block in extracted_blocks:
        section = block.get("section", "").strip() or "Unknown"
        if section.strip().title() in excluded_sections:
            continue
        section_block = build_section_block(section, block.get("text", ""), paper, page, len(blocks))
        if section_block:
            blocks.append(section_block)
    return blocks


def extract_blocks_from_paper(
    paper: dict[str, Any],
    excluded_sections: set[str] | None = None,
) -> list[dict[str, Any]]:
    paper_blocks = []
    for page in paper.get("pages", []):
        page_blocks = extract_blocks_from_page(page, paper=paper, excluded_sections=excluded_sections)
        if page_blocks:
            paper_blocks.extend(page_blocks)
        elif paper_blocks:
            continuation_text = clean_text(remove_known_running_headers(page.get("text") or page.get("raw_text", "")))
            if continuation_text:
                paper_blocks[-1]["text"] = f"{paper_blocks[-1]['text']} {continuation_text}".strip()
        else:
            fallback_block = build_section_block(
                section="Unknown",
                text=page.get("text") or page.get("raw_text", ""),
                paper=paper,
                page=page,
                page_block_index=0,
            )
            if fallback_block:
                paper_blocks.append(fallback_block)
    return paper_blocks


def extract_blocks_from_papers(
    cleaned_papers: list[dict[str, Any]],
    excluded_sections: set[str] | None = None,
) -> list[dict[str, Any]]:
    all_blocks = []
    for paper in cleaned_papers:
        all_blocks.extend(extract_blocks_from_paper(paper, excluded_sections=excluded_sections))
    return all_blocks


def build_text_splitter(chunk_size: int = 500, chunk_overlap: int = 200) -> Any:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )


def chunk_section_blocks(
    blocks: list[dict[str, Any]],
    text_splitter: Any = None,
    starting_chunk_index: int = 0,
) -> list[dict[str, Any]]:
    text_splitter = text_splitter or build_text_splitter()
    chunks = []
    global_chunk_index = starting_chunk_index
    for block in blocks:
        block_text = block.get("text", "")
        if not block_text.strip():
            continue
        splits = text_splitter.split_text(block_text)
        for page_chunk_index, split_text in enumerate(splits):
            split_text = re.sub(r"^[\s\.,;:-]+", "", split_text).strip()
            if split_text:
                metadata = {
                    **block.get("metadata", {}),
                    "section": block.get("section", "Unknown"),
                    "page_chunk_index": page_chunk_index,
                    "chunk_index": global_chunk_index,
                }
                chunks.append({"text": split_text, "metadata": metadata})
                global_chunk_index += 1
    return chunks


def chunk_cleaned_papers(
    cleaned_papers: list[dict[str, Any]],
    text_splitter: Any = None,
    excluded_sections: set[str] | None = None,
) -> list[dict[str, Any]]:
    blocks = extract_blocks_from_papers(cleaned_papers, excluded_sections=excluded_sections)
    return chunk_section_blocks(blocks, text_splitter=text_splitter)


def text_for_ner(chunk: dict[str, Any]) -> str:
    text = chunk.get("text") or ""
    metadata = chunk.get("metadata", {})
    text = re.sub(r"[*_`~^-]+", " ", text)
    text = re.sub(r"^[\s\.,;:-]+", "", text)
    if metadata.get("page_number") == 1:
        title = metadata.get("title") or ""
        if title:
            text = re.sub(re.escape(title), " ", text, flags=re.IGNORECASE)
        authors = metadata.get("authors", "")
        author_list = [a.strip() for a in authors.split(",") if a.strip()] if isinstance(authors, str) else []
        for author in author_list:
            text = re.sub(rf"\b{re.escape(author)}\b", " ", text, flags=re.IGNORECASE)
        parts = re.split(r"\babstract\b[:\-\s]*", text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            text = parts[1]
    return re.sub(r"\s+", " ", text).strip()[:400]


def clean_entities(raw_entities: list[tuple[str, str]]) -> list[tuple[str, str]]:
    cleaned = []
    seen = set()
    for ent_text, ent_label in raw_entities:
        ent_text = ent_text.strip()
        if not ent_text or len(ent_text) < 3:
            continue
        if re.fullmatch(r"\d+(\.\d+)?", ent_text):
            continue
        if ent_label in {"CARDINAL", "ORDINAL"}:
            continue
        if re.fullmatch(r"[^\w]+", ent_text):
            continue
        key = (ent_text.lower(), ent_label)
        if key not in seen:
            seen.add(key)
            cleaned.append((ent_text, ent_label))
    return cleaned


def enrich_chunks_with_entities(
    chunks: list[dict[str, Any]],
    worker_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    worker_path = worker_path or Path(__file__).with_name("ner_worker.py")
    ner_texts = [text_for_ner(chunk) if len(chunk.get("text", "")) > 80 else "" for chunk in chunks]
    proc = subprocess.run(
        [sys.executable, str(worker_path)],
        input=json.dumps(ner_texts),
        text=True,
        capture_output=True,
        check=True,
    )
    all_entities = json.loads(proc.stdout)
    for chunk, raw_entities, ner_text in zip(chunks, all_entities, ner_texts):
        cleaned = clean_entities(raw_entities)
        chunk["metadata"]["entities"] = ", ".join(text for text, _ in cleaned)
        chunk["metadata"]["entity_labels"] = ", ".join(label for _, label in cleaned)
        chunk["metadata"]["ner_text"] = ner_text
    return chunks


def build_final_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_chunks = []
    for index, chunk in enumerate(chunks):
        metadata = {key: chunk["metadata"].get(key, "") for key in METADATA_KEYS}
        if not metadata["paper_id"]:
            metadata["paper_id"] = "paper_0"
        if metadata["chunk_index"] == "":
            metadata["chunk_index"] = index
        final_chunks.append({"text": chunk["text"], "metadata": metadata})
    return final_chunks


class KnowledgeMiningPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        openai_client: Any = None,
        hf_cache: HuggingFacePdfCache | None = None,
    ):
        from openai import OpenAI

        load_dotenv_if_available()
        self.config = config or PipelineConfig()
        self.openai_client = openai_client or OpenAI(api_key=get_openai_api_key())
        self.hf_cache = hf_cache or HuggingFacePdfCache()
        self._chroma_client = None
        self._collection = None
        self._tpm_lock = threading.Lock()
        self._tpm_tokens_used = 0
        self._tpm_window_start = time.monotonic()

    @property
    def chroma_client(self):
        if self._chroma_client is None:
            import chromadb

            self._chroma_client = chromadb.PersistentClient(path=self.config.chroma_path)
        return self._chroma_client

    def get_collection(self, reset: bool = False):
        if reset:
            try:
                self.chroma_client.delete_collection(self.config.collection_name)
            except Exception:
                pass
            self._collection = None
        if self._collection is None:
            self._collection = self.chroma_client.get_or_create_collection(
                name=self.config.collection_name,
                configuration={"hnsw": {"space": "cosine"}},
            )
        return self._collection

    def fetch_papers(self) -> list[dict[str, Any]]:
        return fetch_all_papers(
            params=self.config.arxiv_params,
            base_url=BASE_URL,
            page_size=self.config.arxiv_page_size,
            delay=self.config.request_delay,
        )

    def process_papers(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        processed = download_and_parse_papers(papers, config=self.config, hf_cache=self.hf_cache)
        return clean_processed_papers(processed, self.config.min_page_length)

    def chunks_from_cleaned_papers(
        self,
        cleaned_papers: list[dict[str, Any]],
        enrich_entities: bool = True,
        excluded_sections: set[str] | None = EXCLUDED_SECTIONS,
    ) -> list[dict[str, Any]]:
        splitter = build_text_splitter(self.config.chunk_size, self.config.chunk_overlap)
        chunks = chunk_cleaned_papers(cleaned_papers, text_splitter=splitter, excluded_sections=excluded_sections)
        if enrich_entities and chunks:
            chunks = enrich_chunks_with_entities(chunks)
        return build_final_chunks(chunks)

    def build_chunks_from_arxiv(self, enrich_entities: bool = True) -> list[dict[str, Any]]:
        papers = self.fetch_papers()
        cleaned_papers = self.process_papers(papers)
        return self.chunks_from_cleaned_papers(cleaned_papers, enrich_entities=enrich_entities)

    def get_embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        response = self.openai_client.embeddings.create(model=model or self.config.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def _wait_for_tpm_budget(self, token_count: int) -> None:
        while True:
            with self._tpm_lock:
                now = time.monotonic()
                elapsed = now - self._tpm_window_start
                if elapsed >= 60:
                    self._tpm_tokens_used = 0
                    self._tpm_window_start = now
                    elapsed = 0
                if self._tpm_tokens_used + token_count <= self.config.tpm_limit:
                    self._tpm_tokens_used += token_count
                    return
                wait = 60 - elapsed
            time.sleep(wait + 0.5)

    def _embed_batch(self, batch_index: int, batch_docs: list[str]) -> tuple[int, list[list[float]]]:
        from openai import RateLimitError

        estimated_tokens = sum(len(text) for text in batch_docs) // 4
        self._wait_for_tpm_budget(estimated_tokens)
        for attempt in range(6):
            try:
                return batch_index, self.get_embeddings(batch_docs)
            except RateLimitError:
                if attempt == 5:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("Embedding retry loop exited unexpectedly.")

    def embed_chunks(self, final_chunks: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]], list[str], list[list[float]]]:
        documents = [chunk["text"] for chunk in final_chunks]
        metadatas = [chunk["metadata"] for chunk in final_chunks]
        ids = [f"{chunk['metadata']['paper_id']}_chunk_{chunk['metadata']['chunk_index']}" for chunk in final_chunks]
        batches = [
            (index // self.config.batch_size, documents[index : index + self.config.batch_size])
            for index in range(0, len(documents), self.config.batch_size)
        ]
        all_embeddings_ordered: list[list[list[float]] | None] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=self.config.max_embed_workers) as executor:
            futures = {executor.submit(self._embed_batch, batch_index, batch_docs): batch_index for batch_index, batch_docs in batches}
            for future in as_completed(futures):
                batch_index, embeddings = future.result()
                all_embeddings_ordered[batch_index] = embeddings
        all_embeddings = [embedding for batch in all_embeddings_ordered if batch for embedding in batch]
        return documents, metadatas, ids, all_embeddings

    def store_chunks(
        self,
        final_chunks: list[dict[str, Any]],
        reset_collection: bool = False,
        upsert_batch_size: int = 2000,
    ) -> int:
        documents, metadatas, ids, embeddings = self.embed_chunks(final_chunks)
        collection = self.get_collection(reset=reset_collection)
        for index in range(0, len(documents), upsert_batch_size):
            collection.upsert(
                ids=ids[index : index + upsert_batch_size],
                documents=documents[index : index + upsert_batch_size],
                metadatas=metadatas[index : index + upsert_batch_size],
                embeddings=embeddings[index : index + upsert_batch_size],
            )
        return len(documents)

    def ingest_pdf_file(
        self,
        pdf_path: str | Path,
        enrich_entities: bool = True,
        reset_collection: bool = False,
    ) -> int:
        paper = paper_from_pdf_path(pdf_path)
        parsed = parse_pdf(paper, pdf_path)
        cleaned_papers = clean_processed_papers([parsed], self.config.min_page_length)
        final_chunks = self.chunks_from_cleaned_papers(cleaned_papers, enrich_entities=enrich_entities)
        if not final_chunks:
            return 0
        return self.store_chunks(final_chunks, reset_collection=reset_collection)

    def retrieve_chunks(self, query: str, n_results: int = 3, filter_category: str | None = None) -> dict[str, Any]:
        query_embedding = self.get_embeddings([query])[0]
        where = {"primary_category": filter_category} if filter_category else None
        return self.get_collection().query(query_embeddings=[query_embedding], n_results=n_results, where=where)

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.25,
        filter_category: str | None = None,
        use_hybrid: bool = False,
        entity_bonus: float = 0.5,
    ) -> list[dict[str, Any]]:
        results = self.retrieve_chunks(query, n_results=n_results, filter_category=filter_category)
        formatted_results = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else []
        for index, doc_text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            chunk_id = ids[index] if index < len(ids) else f"chunk_{index}"
            cosine_distance = distances[index] if index < len(distances) else None
            cosine_similarity = 1 - cosine_distance if cosine_distance is not None else None
            if cosine_similarity is not None and cosine_similarity < min_similarity:
                continue
            entities = metadata.get("entities", "")
            entity_overlap = compute_entity_overlap(entities, query)
            hybrid_score = compute_hybrid_score(cosine_similarity, entity_overlap, use_hybrid, entity_bonus)
            formatted_results.append(
                {
                    "rank": index + 1,
                    "chunk_id": chunk_id,
                    "cosine_distance": cosine_distance,
                    "cosine_similarity": cosine_similarity,
                    "hybrid_score": hybrid_score,
                    "entity_overlap": entity_overlap,
                    "text": doc_text,
                    "paper_id": metadata.get("paper_id", ""),
                    "title": metadata.get("title", ""),
                    "section": metadata.get("section", ""),
                    "primary_category": metadata.get("primary_category", ""),
                    "categories": metadata.get("categories", ""),
                    "authors": metadata.get("authors", ""),
                    "published": metadata.get("published", ""),
                    "page_number": metadata.get("page_number", ""),
                    "page_block_index": metadata.get("page_block_index", ""),
                    "page_chunk_index": metadata.get("page_chunk_index", ""),
                    "chunk_index": metadata.get("chunk_index", ""),
                    "entities": entities,
                    "entity_labels": metadata.get("entity_labels", ""),
                }
            )
        if use_hybrid:
            formatted_results = sorted(
                formatted_results,
                key=lambda result: result["hybrid_score"] if result["hybrid_score"] is not None else -1,
                reverse=True,
            )
            for rank, result in enumerate(formatted_results, start=1):
                result["rank"] = rank
        return formatted_results

    def generate_answer(self, query: str, context: str, max_tokens: int = 200, temperature: float = 0.2) -> str:
        response = self.openai_client.chat.completions.create(
            model=self.config.rag_model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Use the retrieved research-paper context below to answer the question.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {query}\n\n"
                        "Answer:"
                    ),
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def rag_query(
        self,
        query: str,
        n_results: int = 3,
        min_similarity: float = 0.25,
        filter_category: str | None = None,
        use_hybrid: bool = False,
        entity_bonus: float = 0.5,
    ) -> dict[str, Any]:
        search_results = self.semantic_search(
            query=query,
            n_results=n_results,
            min_similarity=min_similarity,
            filter_category=filter_category,
            use_hybrid=use_hybrid,
            entity_bonus=entity_bonus,
        )
        if not search_results:
            return {
                "query": query,
                "answer": "I cannot answer this question based on the provided context. No relevant context was found.",
                "sources": [],
            }
        context_parts = []
        sources = []
        for index, result in enumerate(search_results):
            context_parts.append(
                f"[Source {index + 1}: {result['title']}, "
                f"Section: {result['section']}, "
                f"Page {result['page_number']}, "
                f"Chunk {result['chunk_index']}]\n"
                f"{result['text']}"
            )
            sources.append(
                {
                    "source_number": index + 1,
                    "chunk_id": result["chunk_id"],
                    "paper_id": result["paper_id"],
                    "title": result["title"],
                    "section": result["section"],
                    "primary_category": result["primary_category"],
                    "categories": result["categories"],
                    "page_number": result["page_number"],
                    "chunk_index": result["chunk_index"],
                    "cosine_similarity": result["cosine_similarity"],
                    "hybrid_score": result["hybrid_score"],
                    "entity_overlap": result["entity_overlap"],
                    "text": result["text"],
                }
            )
        answer = self.generate_answer(query, "\n\n".join(context_parts))
        return {"query": query, "answer": answer, "sources": sources}


def normalize_entity_set(entities: str) -> set[str]:
    return {entity.strip().lower() for entity in entities.split(",") if entity.strip()}


def compute_entity_overlap(entities: str, query: str) -> int:
    entity_set = normalize_entity_set(entities)
    query_terms = {term.lower().strip() for term in query.split() if term.strip()}
    return len(entity_set.intersection(query_terms))


def compute_hybrid_score(
    cosine_similarity: float | None,
    entity_overlap: int,
    use_hybrid: bool = False,
    entity_bonus: float = 0.5,
) -> float | None:
    if cosine_similarity is None:
        return None
    if not use_hybrid:
        return cosine_similarity
    return cosine_similarity + (entity_overlap * entity_bonus)
