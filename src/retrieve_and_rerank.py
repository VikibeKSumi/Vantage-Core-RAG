from typing import List
from llama_index.core.schema import NodeWithScore



class RetrieveAndRerank:
    """Two-stage retriever: fast vector search + cross-encoder reranking."""

    def __init__(self, ai_core):
        self.ai = ai_core

    def retrieve_and_rerank(self, query: str, index, top_k: int = 4) -> List[NodeWithScore]:
        """
            Performs retrieval + reranking:
            1. Preprocess Indic text
            2. Fast vector retrieval
            3. Cross-encoder reranking
        """
        # Pre-process
        query = self.ai.preprocess_indic_text(query)

        # Stage 1: Fast retrieval
        retriever = index.as_retriever(similarity_top_k=top_k)
        results: List[NodeWithScore] = retriever.retrieve(query)

        if not results:
            print("⚠️ No results found in vector database.")
            return []

        # Stage 2: Rerank
        scores = self.ai.compute_rerank_scores(query, results)

        for i, node in enumerate(results):
            node.score = float(scores[i])

        return sorted(results, key=lambda x: x.score, reverse=True)