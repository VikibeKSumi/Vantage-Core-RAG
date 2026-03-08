# src/engine_load.py
import os
from dotenv import load_dotenv

import torch
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from .database import VectorDBManager
from .ai_core import AICore
from .retrieve_and_rerank import SemanticSearcher
from .generation import Generator
from .config import Config   
from .logger import logger
from .cache import SemanticCache
from .compressor import ContextCompressor


class RAGEngine:
    """Central engine — loads everything ONCE, shares embedding model."""

    def __init__(self):
        try:
            load_dotenv()
            logger.info("🚀 Starting Vantage Core RAG Engine...")

            # Centralized config
            self.config = Config()

            # Auto-detect GPU
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"   • Using device: {self.device.upper()}")

            self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError("❌ GROQ_API_KEY not found in .env")

            # 1. Database + Index
            print("   • Connecting to Qdrant...")
            self.db_manager = VectorDBManager(self.config)   # still passes raw dict for now

            # 2. Load embedding model ONCE — LlamaIndex loads it, we reuse the exact same instance
            print("   • Loading embedding model once (shared with reranker)...")
            
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.config.models["embedding"],
                device=self.device
            )
            
            # Extract the internal SentenceTransformer that LlamaIndex just loaded
            # → Zero duplicate loading
            self.bi_encoder = self.embed_model._model

            self.index = VectorStoreIndex.from_vector_store(
                self.db_manager.get_vector_store(),
                embed_model=self.embed_model
            )

            # 3. AI Core (receives the shared bi_encoder)
            logger.info("   • Loading reranker + Indic normalizer...")
            self.ai_core = AICore(self.config, bi_encoder=self.bi_encoder, device=self.device)
            self.searcher = SemanticSearcher(self.ai_core)

            # 4. Generator
            self.generator = Generator(config=self.config)

            # Semantic Cache (added for production speed)
            self.cache = SemanticCache(similarity_threshold=0.85)
            logger.info("   • Semantic Cache initialized")
            
            # Context Compressor
            self.compressor = ContextCompressor()
            logger.info("   • Context Compressor initialized")

            logger.success("✅ Engine is READY!")

        except Exception as e:
            logger.critical(f"Failed to start engine: {e}", exc_info=True)
            raise RuntimeError("Engine failed to initialize. Check logs/vantage_core.log") from e


    def rewrite_query(self, query: str) -> str:
        """HyDE-style query rewriting for better retrieval."""
        try:
            rewrite_prompt = f"""
            You are an expert at rewriting questions for better retrieval.
            Rewrite this question to be more precise and detailed for vector search.
            Keep the original meaning. Return ONLY the rewritten question.

            Original: {query}

            Rewritten question:
            """

            response = self.generator.llm.complete(rewrite_prompt)
            rewritten = response.text.strip()
            logger.info(f"Original: {query} → Rewritten: {rewritten}")
            return rewritten
        except:
            logger.warning("Query rewrite failed, using original")
            return query
        
    def ask(self, query: str, verbose: bool = True):
        """Main query flow with detailed metrics: time + GPU VRAM usage."""
        import time
        start_total = time.time()


        try:
            # === SEMANTIC CACHE CHECK (first thing we do) ===
            cached = self.cache.get(query)
            if cached:
                if verbose:
                    logger.info("🚀 Semantic Cache HIT — returning instantly")
                    print(f"\n📦 CACHE HIT — Answer returned in 0.00s")
                return {
                    "success": True,
                    "total_latency": 0.0,
                    "cache_hit": True,
                    "answer": cached["answer"],
                    "metrics": cached["metrics"]
                }

     
            # Normal flow continues if cache miss
            if verbose:
                logger.info(f"\n🔍 Query received: {query}")

            # HyDE Query Rewriting (improves retrieval quality)
            rewritten_query = self.rewrite_query(query)
            if verbose and rewritten_query != query:
                logger.info(f"→ Rewritten query: {rewritten_query}")
                
            # 1. Retrieval + Rerank (use rewritten query)
            t0 = time.time()
            retrieved_nodes = self.searcher.retrieve_and_rerank(rewritten_query, self.index)
            retrieval_time = time.time() - t0

            if not retrieved_nodes:
                if verbose:
                    logger.warning("❌ No relevant documents found")
                return {
                    "success": False,
                    "total_latency": round(time.time() - start_total, 2),
                    "empty_retrieval": True,
                    "docs_retrieved": 0
                }

            # Calculate Average Rerank Score (very important quality metric)
            scores = [node.score for node in retrieved_nodes]
            avg_rerank_score = sum(scores) / len(scores)

            top = retrieved_nodes[0]

            if verbose:
                logger.info(f"✅ Top match (score: {top.score:.4f}) | Avg Score: {avg_rerank_score:.4f} | Source: {top.node.metadata.get('file_name', 'Unknown')}")

            # 2. Context Compression (reduces tokens sent to Groq)
            compressed_nodes = self.compressor.compress(retrieved_nodes, max_tokens=1500)
        
            # 3. Generation
            t1 = time.time()
            if verbose:
                logger.info(f"Generating response... (compressed from {len(retrieved_nodes)} to {len(compressed_nodes)} nodes)")
            
            result = self.generator.generate_response(query, compressed_nodes)
            
            
            gen_time = time.time() - t1
            total_time = time.time() - start_total

            # GPU Memory Metrics (only if using GPU)
            vram = {}
            if self.device == "cuda" and torch.cuda.is_available():
                vram = {
                    "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
                    "current_vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 1)
                }

            metrics = {
                "success": True,
                "total_latency": round(total_time, 2),
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": round(gen_time, 2),
                "docs_retrieved": len(retrieved_nodes),
                "top_rerank_score": round(top.score, 4),
                "avg_rerank_score": round(avg_rerank_score, 4),
                "empty_retrieval": False,
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "total_tokens": result.get("total_tokens", 0),
                "tokens_per_second": result.get("tokens_per_second", 0),
                "answer": result["answer"],      
                **vram
            }
            
            if verbose:
                print(f"\n📊 METRICS:")
                print(f"   Retrieval+Rerank : {metrics['retrieval_time']:.2f}s")
                print(f"   Generation       : {metrics['generation_time']:.2f}s")
                print(f"   Total latency    : {metrics['total_latency']:.2f}s")
                print(f"   Docs retrieved   : {metrics['docs_retrieved']}")
                print(f"   Top rerank score : {metrics['top_rerank_score']}")
                print(f"   Avg rerank score : {metrics['avg_rerank_score']}")
                print(f"   Tokens (in/out)  : {metrics['input_tokens']}/{metrics['output_tokens']}")
                print(f"   Tokens/sec       : {metrics['tokens_per_second']}")
                if vram:
                    print(f"   Peak VRAM        : {vram['peak_vram_mb']} MB")
                
                logger.info(f"Response generated | Latency: {metrics['total_latency']}s | Tokens: {metrics['total_tokens']}")
                print(f"\n📝 ANSWER:\n{result['answer']}\n")
                print("-" * 80)

            # Always attach these for UI + future use
            metrics["answer"] = result["answer"]
            metrics["retrieved_nodes"] = retrieved_nodes

            # Record for observability API (production-grade)
            self.query_count = getattr(self, 'query_count', 0) + 1
            self.last_latency = metrics["total_latency"]
            self.last_peak_vram = metrics.get("peak_vram_mb", 0)

            # Store successful response in Semantic Cache for future queries
            self.cache.store(query, result["answer"], metrics)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Unexpected error processing query: {str(e)}", exc_info=True)
            if verbose:
                print("❌ An unexpected error occurred. Please try again.")
            return {
                "success": False,
                "total_latency": round(time.time() - start_total, 2),
                "error": str(e)
            }
    

