from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List


class Embedder():
    def __init__(self, bi_encoder: HuggingFaceEmbedding):
        self.bi_encoder = bi_encoder
    
    def encode(self, query: List[str]):
        
        return self.bi_encoder.get_text_embedding_batch(
            query,
        )
        