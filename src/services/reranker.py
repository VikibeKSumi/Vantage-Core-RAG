from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore
from typing import List

class Reranker():

    def __init__(self, cross_encoder: CrossEncoder):
        self.model = cross_encoder
        

    def rerank(self, query: str, documents: List[NodeWithScore]):
        pairs = [[query, doc.text] for doc in documents]
        return self.model.predict(pairs)