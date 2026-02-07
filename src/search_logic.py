from typing import TYPE_CHECKING

# This prevents circular imports while still allowing the code to run
if TYPE_CHECKING:
    from src.ai_core import AICore

class SemanticSearcher:
    def __init__(self, ai_core):
        self.ai = ai_core

    def search_and_rank(self, query: str, index, top_k=10):
        # 1. AI Pre-processing
        query = self.ai.preprocess_indic_text(query)
        
        # 2. Stage 1: Fast Vector Search (using the index object)
        retriever = index.as_retriever(similarity_top_k=top_k)
        initial_results = retriever.retrieve(query)
        
        if not initial_results:
            print("⚠️ No results found in the vector database.")
            return []

        # 3. Stage 2: Precision Re-ranking
        scores = self.ai.compute_rerank_scores(query, initial_results)
        
        for idx, result in enumerate(initial_results):
            result.score = scores[idx]
            
        return sorted(initial_results, key=lambda x: x.score, reverse=True)
