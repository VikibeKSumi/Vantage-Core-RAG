
import os
import torch
import time
from pathlib import Path
from typing import Dict


from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

from config.config import config

from .services.vector_store import VectorDBManager
from .services.llm import LLMService

from .pipeline.query_rewriter import QueryRewriter
from .pipeline.cache import SemanticCache
from .pipeline.retrieval import Retriever
from .pipeline.reranker import Reranker
from .pipeline.compression import ContextCompressor


from loguru import logger

from dotenv import load_dotenv
load_dotenv()


class Engine():
    
    def __init__(self):
        logger.info("......RAG Engine is On...........")
        self.config = config
        self.embedding_model_name = self.config.models.get("embedding")
        self.reranker_model_name = self.config.models.get("reranker")
        self.device = "cpu" #if torch.cuda.is_available() else "cpu"
        self.db_path = str(Path(self.config.database.get('db_path')))
        self.db_collection_name = self.config.database.get("collection_name")
        self.llm_model_name = self.config.models.get("llm")
        self.api_key = os.getenv("GROQ_API_KEY")


        self.cache_similarity_threshold = 0.92

        logger.info(f"embedding model loading....")
        self.embedding_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
            device=self.device
        )
        logger.info(f"reranking model loading....")
        self.reranking_model = SentenceTransformerRerank(
            model=self.reranker_model_name,
            top_n=4,
            device=self.device
        )

    
        self.query_rewritter = QueryRewriter(api_key=self.api_key, model_name=self.llm_model_name)
        self.semantic_cache = SemanticCache(embedding_model=self.embedding_model, cache_similarity_threshold=self.cache_similarity_threshold)
        self.vector_store = VectorDBManager(db_path=self.db_path, collection_name=self.db_collection_name)
        self.retriever = Retriever()
        self.reranker = Reranker(reranking_model=self.reranking_model)
        self.compression = ContextCompressor()
        self.generator = LLMService(llm_model=self.llm_model_name, api_key=self.api_key)

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store.get_vector_store(),
            embed_model=self.embedding_model
        )
        

    def run(self, query: str, eval_report: bool = False) -> Dict:
        if eval_report:
            logger.info(f"Evaluation is running....")
        logger.info(f"running a query")
        logger.info(f"running on {self.device}")

        logger.info(f"rewritting query....")
        t0 = time.perf_counter()
        rewritten_query = self.query_rewritter.rewrite(query=query)
        
        if not eval_report:
            is_cache, cache_return = self.semantic_cache.get(rewritten_query)
            
            if is_cache:
                logger.info(f"cach hit: returning instant response....")
                info = cache_return
                return info
        
      
        logger.info(f"retrieving from index....")
        t1 = time.perf_counter()
        retrieved_response = self.retriever.retrieve(
            query=rewritten_query,
            index=self.index,
            top_k=20
        )
        response = retrieved_response
        retrieval_time = round(time.perf_counter()-t1, 2)

    
        logger.info(f"reranking retrieved responses....")
        reranked_response = self.reranker.rerank(
            query=rewritten_query,
            retrieved_response=retrieved_response,
        )
        response = reranked_response
            
        
        logger.info(f"compressing reranked response....")
        compressed_response = self.compression.compress(response)

        logger.info(f"generating response....")
        t2 = time.perf_counter()
        result = self.generator.generate_response(rewritten_query, compressed_response)
        generation_time = round(time.perf_counter()-t2, 2)
        total_latency = round(time.perf_counter()-t0, 2)

        result['retrieval_time'] = retrieval_time
        result['generation_time'] = generation_time
        result['total_latency'] = total_latency
        result['cache_hit'] = False

        if not eval_report:
            embedded_query = cache_return

            self.semantic_cache.store(
                query=rewritten_query, 
                result=result,
                embedded_query=embedded_query,
            )
            return result
        
        if eval_report:
            answer = result.get("answer")
            context = [node.node.get_content() for node in compressed_response]
            return {"answer": answer, "contexts": context}
        
        

        
