# FORCE OFFLINE MODE: This prevents the timeout by skipping the HF Hub check
#os.environ['TRANSFORMERS_OFFLINE'] = '1'
#os.environ['HF_DATASETS_OFFLINE'] = '1'

import os
import yaml
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.ai_core import AICore
from src.database import VectorDBManager
from src.search_logic import SemanticSearcher
from src.generation import Generator

# Load variables from .env
load_dotenv()

def main():
    # Load config + API key
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found. Check your .env file.")
        return
    

    
    #1. Load vector index from Qdrant (Load from disk)
    db_manager = VectorDBManager(config)

    llama_embed_model = HuggingFaceEmbedding(
        model_name=config['models']['embedding'],
        device="cpu")
    
    index = VectorStoreIndex.from_vector_store(
        db_manager.get_vector_store(),
        embed_model=llama_embed_model
    )
    
    #2. Run interactive query
    print("\n--- Akshara Strategic Engine Online ---")
    query = input("Type your question: ")

    
    #3. Retrieve + re-rank
    ai_core = AICore(config)
    searcher = SemanticSearcher(ai_core)
    retrieved_nodes = searcher.search_and_rank(query, index)
    

    #4. Generate answer
    if retrieved_nodes:
        print(f"\n✅ [RETRIEVAL SUCCESS] Found {len(retrieved_nodes)} relevant context(s).")
        
        # Print the Top Match specifically
        top_match = retrieved_nodes[0]
        print(f"\n--- TOP CONTEXT (Score: {top_match.score:.4f}) ---")
        print(f"TEXT: {top_match.text}")
        print(f"SOURCE: {top_match.metadata}")
        print("-" * 40)

        print("\n[Thinking...]")
        generator = Generator(api_key=api_key)
        final_answer = generator.generate_response(query, retrieved_nodes)
        print(f"\nADVISOR RESPONSE:\n{final_answer}")
        
    else:
        print("I have no data in the repository to answer that.")

if __name__ == "__main__":
    main()