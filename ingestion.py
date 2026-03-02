# ingestion.py
import sys
from pathlib import Path

# Add src to Python path so we can import our package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.database import VectorDBManager


def main():
    print("🚀 Starting Ingestion Pipeline...")

    config = Config()

    # 1. Load raw documents
    print("   • Loading documents from ./data/raw_docs")
    reader = SimpleDirectoryReader(
        input_dir="./data/raw_docs",
        recursive=True
    )
    documents = reader.load_data()
    print(f"   Loaded {len(documents)} document pages.")

    # 2. Chunking
    parser = SentenceSplitter(
        chunk_size=config.ingestion["chunk_size"],
        chunk_overlap=config.ingestion["overlap"]
    )
    nodes = parser.get_nodes_from_documents(documents)
    print(f"   Created {len(nodes)} chunks.")

    # 3. Database + Index
    print("   • Connecting to Qdrant...")
    db_manager = VectorDBManager(config)        # pass raw dict for now (same as engine)

    print("   • Loading embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name=config.models["embedding"],
        device="cpu"
    )

    storage_context = db_manager.get_storage_context()

    print("   • Building vector index...")
    _ = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )

    print("✅ Ingestion completed successfully!")


if __name__ == "__main__":
    main()