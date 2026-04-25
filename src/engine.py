
import os
import torch
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

from .config.config import config

from .services.vector_store import VectorDBManager
from .services.embedder import Embedder
from .services.llm import LLMService

from .core.text_utils import TextUtils

from .pipeline.cache import SemanticCache
from .pipeline.compression import ContextCompressor
from .pipeline.retrieval import Retriever
from .pipeline.reranker import Reranker

from loguru import logger

from dotenv import load_dotenv
load_dotenv()


class Engine():
    
    def __init__(self):
        logger.info("......RAG Engine is On...........")
        self.config = config
        self.embedding_model_name = self.config.models.get("embedding")
        self.reranker_model_name = self.config.models.get("reranker")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db_path = str(Path(self.config.database.get('db_path')))
        self.db_collection_name = self.config.database.get("collection_name")
        self.llm_model = self.config.models.get("llm")
        self.api_key = os.getenv("GROQ_API_KEY")


        self.similarity_threshold = 0.85

        logger.info(f"embedding model loading....")
        self.embedding_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
            device=self.device
        )
        logger.info(f"reranking model loading....")
        self.reranking_model = SentenceTransformerRerank(
            model=self.reranker_model_name,
            top_n=4
        )

        
        self.embedder = Embedder(embedding_model=self.embedding_model)
        self.retriever = Retriever()
        self.reranker = Reranker(reranking_model=self.reranking_model)
        self.vector_store = VectorDBManager(db_path=self.db_path, collection_name=self.db_collection_name)
        self.llm = LLMService(llm_model=self.llm_model, api_key=self.api_key)

        self.semantic_cache = SemanticCache(embedder=self.embedder, similarity_threshold=self.similarity_threshold)
        self.compression = ContextCompressor()

        
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store.get_vector_store(),
            embed_model=self.embedding_model
        )

    def run(self, query: str):
        logger.info(f"running a query")
        #is_cache = self.semantic_cache.get(query)
        
        #if is_cache:
        #   return is_cache
        
        logger.info(f"retrieving from index....")
        retrieved_response = self.retriever.retrieve(
            query=query,
            index=self.index,
            top_k=4
        )
        logger.info(f"reranking retrieved responses....")
        reranked_response = self.reranker.rerank(
            query=query,
            retrieved_response=retrieved_response,
        )

        logger.info(f"compressing reranked response....")
        compressed_response = self.compression.compress(reranked_response)

        logger.info(f"generating response....")
        result = self.llm.generate_response(query, compressed_response)

        print("-------OUTPUT--------")
        print(result)
        return result

        
