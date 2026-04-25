# src/ai_core.py
from sentence_transformers import CrossEncoder
from indicnlp.normalize.indic_normalize import DevanagariNormalizer


class AICore:
    def __init__(self, config, bi_encoder, device: str = "cpu"):
        """Accept dynamic device (cuda or cpu) from engine."""
        self.device = device
        print(f"--- Loading AI Models on {self.device.upper()} ---")
        
        self.bi_encoder = bi_encoder
        self.cross_encoder = CrossEncoder(config.models["reranker"], device=self.device)
        
        self.normalizer = DevanagariNormalizer()

        
    def preprocess_indic_text(self, text: str) -> str:
        return self.normalizer.normalize(text)

    def get_embeddings(self, text_list: list):
        return self.bi_encoder.encode(text_list, convert_to_tensor=True, show_progress_bar=False)

    def compute_rerank_scores(self, query: str, documents: list):
        pairs = [[query, doc.text] for doc in documents]
        return self.cross_encoder.predict(pairs)