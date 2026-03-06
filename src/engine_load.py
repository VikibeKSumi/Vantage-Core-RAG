# src/engine_load.py
import os
from dotenv import load_dotenv

import torch
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

from .database import VectorDBManager
from .ai_core import AICore
from .retrieve_and_rerank import SemanticSearcher
from .generation import Generator
from .config import Config   


class RAGEngine:
    """Central engine — loads everything ONCE, shares embedding model."""

    def __init__(self):
        load_dotenv()
        print("🚀 Starting Vantage Core RAG Engine...")

        # Centralized config
        self.config = Config()

        # Auto-detect GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   • Using device: {self.device.upper()}")

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
        print("   • Loading reranker + Indic normalizer...")
        self.ai_core = AICore(self.config, bi_encoder=self.bi_encoder, device=self.device)
        self.searcher = SemanticSearcher(self.ai_core)

        # 4. Generator
        self.generator = Generator(config=self.config)

        print("✅ Engine is READY! Ask as many questions as you want.\n")

    def ask(self, query: str, verbose: bool = True):
        """Main query flow with detailed metrics: time + GPU VRAM usage."""
        import time
        import time
        start_total = time.time()

        if verbose:
            print(f"\n🔍 Query: {query}")

        # 1. Retrieval + Rerank
        t0 = time.time()
        retrieved_nodes = self.searcher.retrieve_and_rerank(query, self.index)
        retrieval_time = time.time() - t0

        if not retrieved_nodes:
            if verbose:
                print("❌ No relevant information found.")
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
            print(f"✅ Top match (score: {top.score:.4f}) | Avg Score: {avg_rerank_score:.4f} | Source: {top.node.metadata.get('file_name', 'Unknown')}")

        # 2. Generation
        t1 = time.time()
        if verbose:
            print("🤖 Thinking...")
        
        result = self.generator.generate_response(query, retrieved_nodes)
        
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
            
        if verbose:
            print(f"\n📝 ANSWER:\n{result['answer']}\n")
            print("-" * 80)

        # Always attach these for UI + future use
        metrics["answer"] = result["answer"]
        metrics["retrieved_nodes"] = retrieved_nodes

        return metrics

    

