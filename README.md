# VantageCoreRAG

**Resource-Optimized • Local-First  • Indic-Ready**  
*Bi-Encoder + Cross-Encoder Two-Stage RAG Engine for Consumer Hardware (4GB VRAM)*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![GPU](https://img.shields.io/badge/GPU-4GB_VRAM-green)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-orange)
![Groq](https://img.shields.io/badge/LLM-Groq-purple)

---

### Why Vantage Core-RAG?

A clean, low-resource Retrieval-Augmented Generation system engineered to run **efficiently on modest consumer hardware** (4GB VRAM + 5.6GB RAM) while delivering **high-quality retrieval**.

It uses a **true two-stage architecture**:
1. **Fast Hybrid Search** (BGE-Small + sparse)
2. **Neural Re-ranking** (BGE-Reranker-Base)

This combination dramatically reduces semantic noise and feeds the LLM only the most relevant context — resulting in more accurate and grounded answers.

---

### ✨ Key Features

### Key Features

- **Two-stage retrieval** (Bi-Encoder + Cross-Encoder) for superior relevance on Indic economic documents
- **Full GPU acceleration** for both embedding and reranking (optimized for 4 GB VRAM)
- **Local persistent Qdrant** vector store with hybrid search enabled
- **Native Indic text support** via Devanagari normalization
- **Comprehensive evaluation suite** with full latency breakdown, token usage, VRAM tracking, and P95 metrics
- **Clean, centralized architecture** with single configuration and zero model duplication
- **Production-ready Docker support**
- **Extremely low resource footprint** — runs comfortably on 5.6 GB RAM + 4 GB VRAM
---

### 🛠 Tech Stack

- **Embedding**: `BAAI/bge-small-en-v1.5`
- **Reranker**: `BAAI/bge-reranker-base`
- **Vector Store**: Qdrant (local)
- **LLM**: Groq (llama-3.1-8b-instant)
- **Framework**: LlamaIndex + Sentence-Transformers

---

### 📈 Current Performance (on 4GB VRAM + 5.6GB RAM)
(data set used is economic survey of India 26-27 and Union Budget FYI27 )
- Average Latency: ~21 seconds
- Retrieval + Rerank: 0.59s
- Peak VRAM: 1.26 GB
- Empty Retrieval Rate: 0%

---


### 🚀 Quick Start

# 1. Clone the repository
```bash
    git clone https://github.com/yourusername/vantage_core_rag.git
    cd vantage_core_rag
```
# 2. Create and activate environment
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

# 7. Run the RAG Engine
```bash
    python run.py
```

# 8. (Optional but recommended) Run full evaluation
```bash
    python run_eval.py
```
Type your questions in the terminal after running python run.py. Type 'exit' to quit.

---

### 📁 Folder Structure

```text
vantage_core_rag/
├── run.py                    # Main CLI
├── ingestion.py              # Ingest documents
├── run_eval.py               # Full evaluation
├── src/
│   ├── engine_load.py        # Core engine (loads once)
│   ├── retrieve_and_rerank.py
│   ├── config.py
│   └── ...
├── config/settings.yaml
├── data/raw_docs/            # ← Put your PDFs here
└── evaluation_results.csv
```

## 🛣 Roadmap

- Contextual compression (reduce tokens)
- Semantic caching
- Query rewriting / HyDE
- Streamlit UI
- Faithfulness evaluation
- Docker optimization
- LangChain and Observability

### 🙏 Acknowledgments
- Assisted and Informed greatly by Grok (xAI
- Pytorch
- LLamaIndex

