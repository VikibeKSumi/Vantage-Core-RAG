# Vantage Core-RAG
### Bi-Encoder Optimized Two-Stage Retrieval Pipeline (Local-First)

Vantage Core-RAG is a high-performance, resource-optimized Retrieval-Augmented Generation (RAG) engine. It is designed to run on consumer-grade hardware (4GB VRAM) while maintaining industrial-grade precision through a two-stage retrieval and re-ranking architecture.

## 🏗️ Architecture: The Two-Stage Advantage
Unlike "Naive RAG" systems that rely on a single search step, Vantage Core-RAG utilizes a sophisticated two-stage pipeline:

1.  **Stage 1: Semantic Retrieval (Bi-Encoder)** - Uses `BGE-Small-en-v1.5` to perform high-speed vector search across a local Qdrant database.
    - Optimized for low memory footprint (133MB) to ensure sub-second retrieval on CPU.
2.  **Stage 2: Neural Re-Ranking (Cross-Encoder)**
    - Employs a `BGE-Reranker-Base` to analyze the top candidates side-by-side with the query.
    - Eliminates "semantic noise" and ensures the LLM only receives the most mathematically relevant context.

## 🛠️ Key Features
- **Hybrid Industrial Ingestion:** Recursive directory parsing for PDFs, Markdown, and Unstructured text using `SimpleDirectoryReader`.
- **Hardware-Aware Design:** Orchestrated to run embedding and re-ranking on CPU/RAM, offloading heavy generation to Groq (Llama 3.3 70B) for ultra-fast inference.
- **Multilingual Ready:** Integrated Unicode normalization for Indic script consistency.
- **Local-First Sovereignty:** All sensitive document embeddings and vector storage remain on the local machine.





## 🚀 Quick Start
1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt

2.  This project uses a document corpus for RAG.
Supported sources:
- PDFs
- Markdown
- Plain text

Example documents are provided in `data/raw_docs/`.

To use your own corpus:
- Delete existing docs in `data/raw_docs/`
- Place documents in `data/raw_docs/`

3. Ingest the data:
    ```bash
    python ingest_data.py
    
4. Ask the question in context:
    ```bash
    python run_engine.py



