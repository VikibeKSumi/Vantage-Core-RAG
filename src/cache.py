from sentence_transformers import SentenceTransformer
import torch
from typing import Dict, Optional, Tuple
import time

class SemanticCache:
    """Semantic cache using vector similarity. Stores query → answer + metrics."""

    def __init__(self, config, similarity_threshold: float = 0.85):
        self.cache: Dict[str, dict] = {}           # query_text -> cached data
        self.threshold = similarity_threshold
        self.embedder = None                       # lazy load to save memory
        self.config = config
    
    def _get_embedder(self):
        embed_model= self.config.models['embedding']
        device = self.config.device['device'] or 'cpu'
        if self.embedder is None:
            self.embedder = SentenceTransformer(model_name_or_path=embed_model, device=device)
        return self.embedder

    def get(self, query: str) -> Optional[dict]:
        """Returns cached result if similar query exists."""
        if not self.cache:
            return None

        embedder = self._get_embedder()
        query_emb = embedder.encode(query, convert_to_tensor=True)

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
        embedder = self._get_embedder()
        embedding = embedder.encode(query, convert_to_tensor=True)

        self.cache[query] = {
            "answer": answer,
            "metrics": metrics,
            "embedding": embedding,
            "timestamp": time.time(),
            "cache_hit": False
        }