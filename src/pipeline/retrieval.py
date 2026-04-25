from typing import List
from llama_index.core.schema import NodeWithScore
from ..services.reranker import Reranker
from ..core.text_utils import TextUtils

class Retriever:
    """Two-stage retriever: fast vector search + cross-encoder reranking."""

    def __init__(self, text_utils: TextUtils, reranker: Reranker):
        self.text_utils = text_utils
        self.reranker = reranker
       

    def retrieve(self, query: str, index, top_k: int = 4) -> List[NodeWithScore]:
   
        # Pre-process
        normalized_query: str = self.text_utils.normalize(query)

        # Stage 1: Fast retrieval
        retriever = index.as_retriever(similarity_top_k=top_k)
        retrieved_results: List[NodeWithScore] = retriever.retrieve(normalized_query)

        # Stage 2: Rerank
        rerank_response = self.reranker.rerank(normalized_query, retrieved_results)

        for i, node in enumerate(retrieved_results):
            node.score = float(rerank_response[i])

        return sorted(retrieved_results, key=lambda x: x.score, reverse=True)