
import os
import torch

from .config.config import config
from sentence_transformers import SentenceTransformer, CrossEncoder

from .services.vector_store import VectorDBManager
from .services.embedder import Embedder
from .services.reranker import Reranker
from .services.llm import LLMService

from .pipeline.cache import SemanticCache
from .pipeline.compression import ContextCompressor
from .pipeline.retrieval import Retriever

from dotenv import load_dotenv
load_dotenv()


class Engine():
    
    def __init__(self):
        self.config = config
        self.embedding_model = self.config.models.get("embedding")
        self.reranker_model = self.config.models.get("reranker")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db_path = self.config.database.get('path')
        self.db_collection_name = self.config.database.get("collection_name")
        self.llm_model = self.config.models.get("llm")
        self.api_key = os.getenv("GROQ_API_KEY")

        self.bi_encoder  = """SentenceTransformer(
            self.embedding_model,
            device=self.device
            )"""
        
        self.cross_encoer = """CrossEncoder(
            self.reranker_model,
            device=self.device
            )"""

        self.embedder = Embedder(bi_encoder=self.embedding_model)
        self.reranker = Reranker(cross_encoder=self.reranker_model)
        self.vector_store = VectorDBManager(db_path=self.db_path, collection_name=self.db_collection_name)
        self.llm = LLMService(llm_model=self.llm_model, api_key=self.api_key)

        """self.cache = SemanticCache()
        self.compression = ContextCompressor()
        self.retrieval = Retriever()"""

    def run(self, query: str):
        print("HERE")
        print(query)


