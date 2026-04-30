# Enterprise Knowledge Mining System

## Overview

The Enterprise Knowledge Mining System is a scalable document processing and question-answering system built on arXiv research papers. It extracts structured knowledge from unstructured PDFs and enables semantic search and Retrieval-Augmented Generation (RAG).

The system processes research papers, builds embeddings, stores them in a vector database, and answers natural language queries grounded in retrieved content.

## Key Features

- Semantic search over research papers
- PDF ingestion and parsing (PyMuPDF)
- Section-aware chunking
- Named Entity Recognition (spaCy `en_core_web_sm`)
- OpenAI embeddings (`text-embedding-3-small`)
- ChromaDB vector storage (cosine similarity)
- RAG-based Q&A (`gpt-4.1-mini`)
- Streamlit UI for querying the system
- RAG evaluation using ground truth + RAGAS

## System Architecture

- **Ingest:** arXiv API / PDFs
- **Process:** Cleaning + Section Extraction
- **Chunk:** Recursive text splitting
- **NER:** Entity extraction via spaCy
- **Embed:** OpenAI embeddings
- **Store:** ChromaDB (vector DB)
- **Retrieve:** Semantic search
- **Answer:** RAG (LLM)

## Project Structure

- data-processingV2.ipynb
- knowledge_pipeline.py
- ner_worker.py
- hf_index_loader.py
- debug_utils.py
- streamlit_app.py
- ground_truth_dataset.json
- chroma_db/
- README.md

## Environment Variables Setup

Add a `.env` file in your project root with the following structure:

- `OPENAI_KEY="your-api-key"`
- `HF_TOKEN=your-hf-token`
- `HF_REPO_ID=your-repo-id`

## Environment Setup

- Use **Python 3.11** (recommended for stability)

# 1. Data Processing Environment

## Create virtual environment

- `py -3.11 -m venv .venv`
- Activate: `./.venv/Scripts/activate`
- Install dependencies: `pip install -r requirements.txt`
- Install spaCy model: `python -m spacy download en_core_web_sm`

### Run Data Processing Pipeline

- Open: `data-processingV2.ipynb`
- This pipeline:
  - Fetches arXiv papers
  - Downloads PDFs
  - Cleans and extracts text
  - Splits into chunks
  - Runs NER
  - Generates embeddings
  - Stores in ChromaDB
- Models used:
  - `embedding_model = "text-embedding-3-small"`
  - `rag_model = "gpt-4.1-mini"`

# 2. Streamlit App (Query Interface)

- Link: https://enterprise-knowledge-mining-system-enterpr-streamlit-app-qwxgr8.streamlit.app
- Run locally: `streamlit run streamlit_app.py`
- Features:
  - Ask natural language questions
  - View retrieved chunks
  - Inspect similarity scores
  - Optional hybrid reranking

# 3. Evaluation Strategy

The system was evaluated using three complementary approaches to ensure both practical correctness and measurable performance.

## 3.1 Custom Evaluation

- Verified relevant chunks appear in top-k results
- Observed cosine similarity range: 0.57 – 0.76
- Correct documents consistently retrieved within top 5
- Top similarity remains stable across k
- Average similarity decreases as k increases
- Confirms weaker results appear at higher k
- **Conclusion:** Optimal retrieval range is k = 3–5
- RAG Answer Behaviour:
  - Answerable → correct grounded answers
  - Partial → partially correct responses
  - Unrelated → correctly returns no context
  - Example: Query: "chocolate cake" → No relevant context returned
- Hybrid Search Evaluation:
  - Compared semantic vs hybrid retrieval
  - Only 1/5 queries changed ranking
  - Entity overlap often zero
  - **Conclusion:** Hybrid reranking currently provides minimal benefit
- NER Quality Analysis:
  - Common entities: CPU, GPU, IEEE, CNN
  - Noise observed: “Fig”, “-1”
  - **Conclusion:** NER useful for metadata, but limited for ranking
- Chunk Quality Analysis:
  - ~85,000+ chunks generated
  - Avg length ~536 characters
  - Most near max size (~800)
  - ~5.9% noise chunks

## 3.2 Ground Truth Evaluation

- Dataset: `ground_truth_dataset.json`
- Contains:
  - Questions
  - Expected answers
  - Source titles
  - Keywords
- Used to:
  - Validate retrieval correctness
  - Compare generated vs expected answers
  - Measure keyword coverage

## 3.3 RAGAS Evaluation

- RAGAS was used for standardized evaluation of:
  - Answer Correctness
  - Faithfulness
  - Context Precision
  - Context Recall
- This evaluates:
  - How well answers are grounded in retrieved context
  - Whether retrieved context is relevant
  - Whether hallucination occurs
- **Why Multiple Evaluation Methods?**
  - Custom evaluation → Understand system behaviour and retrieval quality
  - Ground truth evaluation → Validate correctness against expected outputs
  - RAGAS evaluation → Provide standardized, quantitative metrics
- Together, these ensure both practical reliability and formal validation.

# 4. RAG Evaluation Setup (RAGAS)

- Create separate environment: `py -3.11 -m venv .venv-ragas`
- Activate: `./.venv-ragas/Scripts/activate`
- Install dependencies: `pip install -r requirements-ragas.txt`

- **RAGAS uses the following models for evaluation:**
  - LLM: `gpt-4o-mini`
  - Embedding model: `text-embedding-3-small`

# Core Pipeline Details

- **PDF Processing:**
  - PyMuPDF extraction
  - Removes artifacts, page markers, hyphenation
- **Chunking:**
  - RecursiveCharacterTextSplitter
  - Chunk size ~800
  - Overlap ~50
- **Entity Extraction:**
  - spaCy (`en_core_web_sm`)
  - Parallelized worker
  - Stored in metadata
- **Embeddings:**
  - OpenAI API
  - Batch embedding with retry handling
- **Storage:**
  - ChromaDB PersistentClient
  - Cosine similarity
- **Retrieval:**
  - Embedding-based semantic search
  - Optional hybrid reranking
- **RAG:**
  - Context from top-k chunks
  - Answer generation using gpt-4.1-mini
- **Evaluation Summary:**
  - Strong retrieval performance
  - Optimal k ≈ 5
  - Hybrid reranking limited impact
  - NER useful but imperfect
  - High-quality chunk distribution
- **Limitations:**
  - No multi-modal support (tables/images)
  - NER misses domain-specific terms
  - Fixed top-k retrieval
  - PDF extraction noise
- **Future Improvements:**
  - Neo4j graph-based knowledge layer
  - Multi-modal ingestion
  - Feedback loop for answers
  - Improved NER models
