from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List


class Embedder():
    def __init__(self, embedding_model: HuggingFaceEmbedding):
        self.embedding_model = embedding_model
    
    def encode(self, query: List[str]):
        
        return self.embedding_model.get_text_embedding_batch(
            [query]
        )
        