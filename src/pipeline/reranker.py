from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SentenceTransformerRerank
from typing import List



class Reranker():

    def __init__(self, reranking_model: SentenceTransformerRerank):
        self.reranking_model = reranking_model
        

    def rerank(self, query: str, retrieved_response: List[NodeWithScore]) -> List[NodeWithScore]:
        return self.reranking_model.postprocess_nodes(retrieved_response, query_str=query)
