import time
import pandas as pd
import os
import sys
import yaml
from dotenv import load_dotenv

# Fix path to find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_core import AICore
from src.search_logic import SemanticSearcher
from src.generation import Generator
from src.database import VectorDBManager
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load Config
load_dotenv()
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

# Test Questions (Add specific questions from your 120-page docs here)
TEST_QUERIES = [
    "What is the primary architectural principle of this system?",
    "How does the two-stage retrieval process work?",
    "What are the hardware constraints defined for this project?",
]

def run_evaluation():
    print("--- 🧪 Starting Vantage Core-RAG Evaluation ---")
    
    # 1. Initialize Components
    print("1. Loading Engine...", end=" ")
    ai_core = AICore(config)
    searcher = SemanticSearcher(ai_core)
    model_name = config['models'].get('llm', 'llama3-70b-8192')
    generator = Generator(api_key=os.getenv("GROQ_API_KEY"), model=model_name)
    
    # Load Index
    db_manager = VectorDBManager(config)
    vector_store = db_manager.get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = HuggingFaceEmbedding(model_name=config['models']['embedding'], device="cpu")
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
    print("✅ Online.")

    results = []

    # 2. Run Loop
    for query in TEST_QUERIES:
        print(f"\n🔎 Testing: '{query}'")
        start_time = time.time()
        
        # A. Search (Stage 1 & 2)
        nodes = searcher.search_and_rank(query, index, top_k=5)
        
        # B. Generate
        if nodes:
            response = generator.generate_response(query, nodes)
            context_score = nodes[0].score if hasattr(nodes[0], 'score') else 0.0
        else:
            response = "No Context Found"
            context_score = 0.0
            
        end_time = time.time()
        latency = round(end_time - start_time, 2)
        
        print(f"   ⏱️ Latency: {latency}s | 🎯 Top Score: {context_score:.4f}")
        
        results.append({
            "Query": query,
            "Latency (s)": latency,
            "Top Re-rank Score": context_score,
            "Response Preview": response[:50] + "..."
        })

    # 3. Save Report
    df = pd.DataFrame(results)
    print("\n--- 📊 Final Report ---")
    print(df.to_markdown(index=False))
    
    # Optional: Save to CSV
    df.to_csv("evaluation_report.csv", index=False)
    print("\n✅ Report saved to 'evaluation_report.csv'")

if __name__ == "__main__":
    run_evaluation()