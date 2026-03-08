# VantageCoreRAG

**Production-Ready Local RAG System**  
*Optimized for Consumer Hardware (4GB VRAM + 5.6GB RAM)*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![GPU](https://img.shields.io/badge/GPU-4GB_VRAM-green)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-orange)
![Groq](https://img.shields.io/badge/LLM-Groq-purple)

---

### Why Vantage Core-RAG?

A clean, low-resource Retrieval-Augmented Generation system engineered to run **efficiently on modest consumer hardware** (4GB VRAM + 5.6GB RAM) while delivering **high-quality retrieval**.

---

### ✨ Key Features

- **Two-stage retrieval** (Bi-Encoder + Cross-Encoder) for superior relevance on Indic economic documents
- **Semantic Cache** – Instant responses for similar/repeated queries (major latency win)
- **Query Rewriting (HyDE)**: Improves retrieval quality by rewriting user questions
- **Context Compression**: Reduces tokens sent to Groq (faster & cheaper generation)
- **Full GPU acceleration** for both embedding and reranking (optimized for 4 GB VRAM)
- **Local persistent Qdrant** vector store with hybrid search enabled
- **Native Indic text support** via Devanagari normalization
- **Production Observability**: FastAPI `/health` + Prometheus-compatible `/metrics` endpoint
- **Structured Logging** with Loguru (console + file)
- **Automatic Retries & Rate Limiting**: Graceful handling of Groq API limits
- **Comprehensive evaluation suite** with full latency breakdown, token usage, VRAM tracking, and P95 metrics
- **Clean, centralized architecture** with single configuration and zero model duplication
- **Streamlit UI** with live metrics display
- **Production-ready Docker support** (Dockerfile + docker-compose ready)
- **Robust error handling** & graceful degradation
- **Extremely low resource footprint** — runs comfortably on 5.6 GB RAM + 4 GB VRAM

---

### 🛠 Tech Stack

- **Embedding**: `BAAI/bge-small-en-v1.5`
- **Reranker**: `BAAI/bge-reranker-base`
- **Vector Store**: Qdrant (local)
- **LLM**: Groq (llama-3.1-8b-instant)
- **Framework**: LlamaIndex + Sentence-Transformers
- **Observability**: FastAPI + Loguru
- **UI**: Streamlit

---

### 📈 Current Performance (Batch Process)
**Under consumer hardware constraints (4GB VRAM + 5.6GB RAM) and Groq API:**

- Average Total Latency: **19.67 seconds**
- Retrieval + Rerank: **0.59 seconds**
- Peak VRAM: **1.18 GB**
- Empty Retrieval Rate: **0.0%**
- Avg Tokens per Query: **2362**

*Note: Latency is dominated by Groq generation. Hybrid local LLM fallback is planned for <12s average.*

---

### 🚀 Quick Start

# 1. Clone the repository
```bash
    git clone https://github.com/yourusername/vantage_core_rag.git
    cd vantage_core_rag
```
# 2. (optional but recommended) Create and activate environment
```bash
    conda create -n vantage python=3.10 -y
    conda activate vantage
```

# 3. Install dependencies
```bash
    pip install -r requirements.txt
```

# 4. Add your Groq API key 
(Create a file named .env in the root folder and put this line inside it)
```env
    GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

# 5. Put your PDF files
(Budget, Economic Survey provided inside this folder)
```text
    data/raw_docs/
```

# 6. Ingest the documents
```bash
    python ingestion.py
```

# 7. (Main) Run the RAG Engine
```bash
    python run.py
```

# 8. (Optional but recommended) Run full evaluation
```bash
    python run_eval.py
```
Type your questions in the terminal after running python run.py. Type 'exit' to quit.

# 9. (Optional but recommended) Run Streamlit UI
```bash
    streamlit run app.py
```
# 10. (Optional but recommended) Run Observability API (in another terminal)
```
    python api.py
```

---

### 📁 Folder Structure

```text
vantage_core_rag/
├── run.py                    # Main CLI entry point
├── app.py                    # Streamlit UI
├── api.py                    # Observability API
├── ingestion.py              # Document ingestion
├── run_eval.py               # Full benchmarking
├── docker-compose.yml
├── src/
│   ├── engine_load.py        # Core orchestration engine
│   ├── retrieve_and_rerank.py
│   ├── cache.py              # Semantic Cache
│   ├── compressor.py         # Context Compression
│   ├── logger.py
│   ├── config.py
│   └── ...
├── config/settings.yaml
└── data/raw_docs/
```

---

### 🚧 Next Priorities (High Impact)
- Hybrid local LLM fallback (biggest latency win)
- Advanced Reranker
- Persistent Semantic Cache (disk-backed)
- Qdrant Server mode (true multi-service support)
- Advanced faithfulness evaluation
- Observability Dashboard

---

### Built With
Assisted and guided by Grok (xAI) • LlamaIndex • Sentence-Transformers • Qdrant • Groq

