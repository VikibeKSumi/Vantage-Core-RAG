import sys
from pathlib import Path

# Add src to Python path so we can import our package
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from src.config.config import config
from src.services.vector_store import VectorDBManager

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from pathlib import Path
from loguru import logger


def ingestion():
    logger.info(f".........Starting Ingestion Pipeline.........")
    raw_data_path = str(Path(config.database.get("raw_data_path")))
    db_path = str(Path(config.database.get("db_path")))
    collection_name = config.database.get("collection_name")
    embed_model_name = config.models.get("embedding")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        device=device
    )

    # 1. Load raw documents
    logger.info(f"Loading documents....")
    reader = SimpleDirectoryReader(
        input_dir=str(raw_data_path),
        recursive=True
    )
    
    documents = reader.load_data()
    logger.info(f"loaded {len(documents)} document pages")

    # 2. Chunking
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=Settings.embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(f"created {len(nodes)} chunks....")

    # 3. Database + Index
    logger.info("connecting to Qdrant...")
    db_manager = VectorDBManager(
        db_path=db_path,
        collection_name=collection_name
    )      

    storage_context = db_manager.get_storage_context()

    logger.info("building vector index....")
    _ = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )

    logger.info(".......Ingestion Successful!!.......")


if __name__ == "__main__":
    ingestion()