import torch
from typing import Dict, Optional
import time
from ..services.embedder import Embedder


class SemanticCache:
    """Semantic cache using vector similarity. Stores query → answer + metrics."""

    def __init__(self, embedder: Embedder, similarity_threshold: float = 0.85):
        self.embedder = embedder  
        self.cache: Dict[str, dict] = {}
        self.threshold = similarity_threshold


    def get(self, query: str) -> Optional[dict]:
        """Returns cached result if similar query exists."""
        if not self.cache:
            return None

        query_emb = self.embedder.encode(query)

        for cached_query, data in self.cache.items():
            cached_emb = data["embedding"]
            similarity = torch.nn.functional.cosine_similarity(query_emb, cached_emb, dim=0)

            if similarity >= self.threshold:
                data["cache_hit"] = True
                data["cache_hit_time"] = time.time()
                return data

        return None

    def store(self, query: str, answer: str, metrics: dict):
        """Store query + answer + metrics in cache."""
        
        embedding = self.embedder.encode(query)


        self.cache[query] = {
            "answer": answer,
            "metrics": metrics,
            "embedding": embedding,
            "timestamp": time.time(),
            "cache_hit": False
        }