import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Dict
import time

class SemanticCache:
    

    def __init__(self, embedding_model: HuggingFaceEmbedding, cache_similarity_threshold: float = 0.85):
        self.embedding_model = embedding_model
        self.cache_similarity_threshold = cache_similarity_threshold
        self.cache: Dict[str, dict] = {}


    def get(self, query: str) -> tuple:

        embedded_query = torch.tensor(
            self.embedding_model.get_text_embedding(
                query
        ))

        if not self.cache:
            # cache is empty
            return None, embedded_query
        
        for cached_query, info in self.cache.items():
            cached_emb = info.get("embedding")
            similarity = torch.nn.functional.cosine_similarity(
                embedded_query,
                cached_emb,
                dim=0
            )

            if similarity >= self.cache_similarity_threshold:
                # cache match
                return True, {**info["result"], "cache_hit":True}


        # cache no match
        return False, embedded_query

    def store(self, query: str, result: dict, embedded_query: torch.tensor):

        self.cache[query] = {
            "result": result,
            "embedding": embedded_query,
            "timestamp": time.time()
        }