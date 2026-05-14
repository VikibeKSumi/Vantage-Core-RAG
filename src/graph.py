import os
import time
from pathlib import Path

import torch
from loguru import logger
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from langgraph.graph import StateGraph,END

from .state import ResponseState
from config.config import config
from .services.vector_store import VectorDBManager
from .services.llm import LLMService
from .pipeline.query_rewriter import QueryRewriter
from .pipeline.cache import SemanticCache
from .pipeline.retrieval import Retriever
from .pipeline.reranker import Reranker
from .pipeline.compression import ContextCompressor


load_dotenv()

class AgenticRAG():

    def __init__(self):
    
        logger.info("......RAG engine is turning on...........")

        self.config = config
        self.embedding_model_name = self.config.models.get("embedding")
        self.reranker_model_name = self.config.models.get("reranker")
        self.db_collection_name = self.config.database.get("collection_name")
        self.llm_model_name = self.config.models.get("llm")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db_path = str(Path(self.config.database.get('db_path')))
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
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store.get_vector_store(),
            embed_model=self.embedding_model
        )
        self.retriever = Retriever(index=self.index, top_k=20)
        self.reranker = Reranker(reranking_model=self.reranking_model)
        self.compression = ContextCompressor()
        self.generator = LLMService(llm_model=self.llm_model_name, api_key=self.api_key)


        self.workflow = StateGraph(ResponseState)
        logger.info("engine is live....")

        self.create_nodes()
        self.create_edges()

        logger.info("compiling graph....")
        self.app = self.workflow.compile()


    def cache_hit_router(self, state: ResponseState):
        cache_hit = state.get("cache_hit")
        return cache_hit
    

    def create_nodes(self):

        logger.info("creating graph nodes....")
        self.workflow.add_node("query_rewriter_node", self.query_rewritter.rewrite)
        self.workflow.add_node("semantic_cache_node", self.semantic_cache.get)
        self.workflow.add_node("retriever_node", self.retriever.retrieve)
        self.workflow.add_node("reranker_node", self.reranker.rerank)
        self.workflow.add_node("context_compressor_node", self.compression.compress)
        self.workflow.add_node("llm_service_node", self.generator.generate_response)

    def create_edges(self):
        logger.info("creating graph edges....")
        self.workflow.set_entry_point("query_rewriter_node")
        self.workflow.add_edge("query_rewriter_node", "semantic_cache_node")
        self.workflow.add_conditional_edges(
            "semantic_cache_node",
            self.cache_hit_router,
            {
                True: END,
                False: "retriever_node"
            })
        self.workflow.add_edge("retriever_node", "reranker_node")
        self.workflow.add_edge("reranker_node", "context_compressor_node")
        self.workflow.add_edge("context_compressor_node", "llm_service_node")
        self.workflow.add_edge("llm_service_node", END)


    def run_graph(self, query):

        logger.info("graph is working on query....")
        user_input = {"query": query}
        result = self.app.invoke(
            input=user_input
        )

        logger.info("graph execution is over...")
        return result


