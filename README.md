# Enterprise Knowledge Mining System

## Project overview

The Enterprise Knowledge Mining System is designed to help organizations extract actionable insights from large collections of unstructured documents, focusing on research papers from arXiv. The system automates information extraction, builds a semantic search engine, and answers natural language questions using retrieval-augmented generation (RAG).

## Current implementation status

The current notebook stores metadata and entity information with each chunk, but retrieval is currently driven only by embedding similarity. Metadata filtering and hybrid retrieval are possible future next steps, but not implemented yet.

### 1. Document ingestion

- arXiv API search is working.
- Paper metadata is extracted, including arXiv ID, title, authors, published date, primary category, category list, summary, and PDF link.
- The arXiv XML namespace is used to extract arXiv-specific metadata such as primary category.
- PDF download workflow is implemented for a specific single download.

### 2. PDF processing

- PDFs are converted to markdown using `pymupdf4llm`.
- Page-level markdown extraction is working.
- Page metadata such as page number, page count, file path, and document properties are available.

### 3. Section-aware preprocessing

- Section blocks are extracted from markdown using heading-based rules (##, \*\*, etc.).
- Text cleaning is implemented for chunk preparation.
- NER-specific text cleaning is separated from general chunk text cleaning.
- Entity post-processing removes duplicates and obvious noise.

### 4. Chunking and metadata enrichment

- Section-aware chunking is implemented.
- Chunk metadata currently includes:
  - paper ID
  - title
  - section
  - primary category
  - category list
  - authors
  - published date
  - page number
  - chunk index
- Chunk metadata is enriched with extracted entities and entity labels.

### 5. Entity extraction

- spaCy NER is integrated using `en_core_web_md` will eventually switch to `en_core_web_sm` when we increase research papers for faster performance.
- Extracted entities are cleaned and attached to chunk metadata.
- NER currently acts as metadata enrichment rather than retrieval ranking logic.

### 6. Embeddings and vector storage

- OpenAI embeddings are generated with `text-embedding-3-small`, will switch to `text-embedding-3-large` when we increase research papers.
- ChromaDB persistent storage is configured and working.
- Collection is configured to use cosine similarity instead of the default Euclidean distance.
- Chunk text, metadata, IDs, and embeddings are being stored successfully.

### 7. Semantic search engine

- Query embedding is generated and compared against stored chunk embeddings.
- Top matching chunks are returned from ChromaDB.
- A minimum similarity threshold is applied to filter weak matches.
- Search results are formatted with academic metadata for inspection.

### 8. RAG pipeline

- RAG is correctly linked to the semantic search component.
- Retrieved chunks are used to build grounded context for the LLM.
- The LLM generates answers based only on retrieved context.
- Source information is returned separately from the answer.
- The current pipeline correctly handles:
  - answerable questions
  - partially answerable questions
  - unrelated questions with insufficient context

### Next Steps:

- refactor arXiv fetching to support batch paper retrieval
- implement batch PDF downloading
- convert pdf to markdown directly in chunk processing
- refactor the notebook into reusable functions and separate modules
  - move helper functions into dedicated files for ingestion, cleaning, processing, vector storage, and RAG etc
- build process_pdf_to_chunks() as the main processing pipeline
- scale from 1 paper to 10–20 papers first
- tune chunk size, overlap, and top-k retrieval
- add logging/status tracking for downloads and processing
- metadata filtering during retrieval
- hybrid retrieval using entity overlap or metadata boosts
- deploy a UI only after multi-paper retrieval is stable; streamlit
