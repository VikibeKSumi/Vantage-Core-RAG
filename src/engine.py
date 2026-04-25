
import os
import torch
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .config.config import config
from sentence_transformers import CrossEncoder

from .services.vector_store import VectorDBManager
from .services.embedder import Embedder
from .services.reranker import Reranker
from .services.llm import LLMService

from .core.text_utils import TextUtils

from .pipeline.cache import SemanticCache
from .pipeline.compression import ContextCompressor
from .pipeline.retrieval import Retriever

from dotenv import load_dotenv
load_dotenv()


class Engine():
    
    def __init__(self):
        self.config = config
        self.embedding_model_name = self.config.models.get("embedding")
        self.reranker_model_name = self.config.models.get("reranker")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db_path = str(Path(self.config.database.get('db_path')))
        self.db_collection_name = self.config.database.get("collection_name")
        self.llm_model = self.config.models.get("llm")
        self.api_key = os.getenv("GROQ_API_KEY")

        self.text_utils = TextUtils()
        self.similarity_threshold = float(0.85)

    
        self.bi_encoder = HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
            device=self.device
        )
        self.cross_encoder = CrossEncoder(
            self.reranker_model_name,
            device=self.device
        )

        
        self.embedder = Embedder(bi_encoder=self.bi_encoder)
        self.reranker = Reranker(cross_encoder=self.cross_encoder)
        self.retriever = Retriever(text_utils=self.text_utils, reranker=self.reranker)
        self.vector_store = VectorDBManager(db_path=self.db_path, collection_name=self.db_collection_name)
        self.llm = LLMService(llm_model=self.llm_model, api_key=self.api_key)

        self.semantic_cache = SemanticCache(embedder=self.embedder, similarity_threshold=self.similarity_threshold)
        self.compression = ContextCompressor()

        
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store.get_vector_store(),
            embed_model=self.bi_encoder
        )

    def run(self, query: str):
        
        #is_cache = self.semantic_cache.get(query)
        
        #if is_cache:
        #   return is_cache
                    
        retrieved_response = self.retriever.retrieve(
            query=query,
            index=self.index,
            top_k=4
        )
        compressed_response = self.compression.compress(retrieved_response)

        result = self.llm.generate_response(query, compressed_response)

        print(result)

        
