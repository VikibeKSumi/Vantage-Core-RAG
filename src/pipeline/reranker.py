from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SentenceTransformerRerank
from typing import List

from ..state import ResponseState

class Reranker():

    def __init__(self, reranking_model: SentenceTransformerRerank):
        self.reranking_model = reranking_model
        

    def rerank(self, state: ResponseState) -> List[NodeWithScore]:
        rewritten_query = state.get("rewritten_query")
        retrieved_nodes = state.get("retrieved_nodes")

        return {"reranked_nodes": self.reranking_model.postprocess_nodes(
            retrieved_nodes, 
            query_str=rewritten_query
        )}
