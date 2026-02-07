# Akshara RAG Engine: Technical Architecture Map

## 📂 Project Structure
- `data/` : Local storage for raw text files and the Qdrant vector database.
- `src/` : The core logic of the AI system.
- `config/` : Configuration files (settings.yaml) for model names and paths.
- `.env` : Secure storage for your API keys (Groq/Gemini).

## 🛠️ Module & Function Breakdown

### 1. The Muscle: `src/ai_core.py` (AksharaAICore)
- `__init__`: Initializes the BGE-Small (Bi-Encoder) and BGE-Reranker (Cross-Encoder) models on the CPU.
- `preprocess_indic_text(text)`: Normalizes Devanagari/Hindi scripts to ensure math vectors are consistent across different typing styles.
- `compute_rerank_scores(query, nodes)`: Compares the query to retrieved results line-by-line to find the absolute best match.

### 2. The Vault: `src/database.py` (VectorDBManager)
- `get_vector_store()`: Connects the engine to the Qdrant local database.
- `get_storage_context()`: Defines the persistent storage path on your hard drive.

### 3. The Eyes: `src/search_logic.py` (SemanticSearcher)
- `search_and_rank(query, index)`: 
    1. Pre-processes the query.
    2. Performs a hybrid vector search.
    3. Triggers the Reranker to pick the top 3-5 high-quality results.

### 4. The Voice: `src/generation.py` (AksharaGenerator)
- `generate_response(query, nodes)`: Takes the raw context from the vault, packages it into a prompt, and sends it to **Groq (Llama 3.3 70B)** to generate a professional "Architect" response.

## 🚀 The Execution Flow
1. **User Query** (run_engine.py) → 
2. **Text Normalization** (ai_core.py) → 
3. **Vector Retrieval** (database.py + search_logic.py) → 
4. **Precision Reranking** (ai_core.py) → 
5. **AI Synthesis** (generation.py via Groq) → 
6. **Final Advisor Response**