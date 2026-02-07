from sentence_transformers import SentenceTransformer, CrossEncoder
from indicnlp.normalize.indic_normalize import DevanagariNormalizer

class AICore:
    def __init__(self, config):
        # FORCE CPU to stabilize the 5.6GB RAM environment
        self.device = "cpu"
        
        print(f"--- Loading AI Models on {self.device} (Memory Optimization Mode) ---")
        
        # Load the Lightweight Bi-Encoder
        self.bi_encoder = SentenceTransformer(config['models']['embedding'], device=self.device)
        
        # Load the Base Re-ranker
        self.cross_encoder = CrossEncoder(config['models']['reranker'], device=self.device)
        # 3. Normalizer for Devanagari scripts (Hindi, Marathi, etc.)
        self.factory = DevanagariNormalizer()
   
    def preprocess_indic_text(self, text: str) -> str:
        """
        AI-specific cleaning: Normalizes unicode characters 
        to ensure 'ि' + 'क' is always seen as 'कि' by the tokenizer.
        """
        return self.factory.normalize(text)

    def get_embeddings(self, text_list: list):
        """Generates dense vectors for the vector database."""
        # We use convert_to_tensor for faster processing on GPU
        return self.bi_encoder.encode(text_list, convert_to_tensor=True, show_progress_bar=False)

    def compute_rerank_scores(self, query: str, documents: list):
        """
        The 'Deep' AI Step: 
        Unlike Cosine Similarity, this looks at token-level interaction.
        """
        pairs = [[query, doc.text] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        return scores