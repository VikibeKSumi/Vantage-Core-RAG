import sys
from pathlib import Path

# Add src to Python path so we can import our package
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import hashlib
import json

from config.config import config
from src.services.vector_store import VectorDBManager

from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from pathlib import Path
from loguru import logger


def ingestion():
    logger.info(f".........Starting Ingestion Pipeline.........")
    raw_docs_path = Path(config.database.get("raw_data_path"))
    hash_data_path = Path("data/hash_store") / "hash_data.json"
    db_path = str(Path(config.database.get("db_path")))
    collection_name = config.database.get("collection_name")
    embed_model_name = config.models.get("embedding")
    device = "cpu" if torch.cuda.is_available() else "cpu"

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        device=device
    )

    parser = PyMuPDFReader()
    file_extractor = {'.pdf': parser}


    logger.info(f"loading hashes....")
    hash_data_path.parent.mkdir(parents=True, exist_ok=True)
    try: 
        with open(hash_data_path, "r") as f:
            hash_set = set(json.load(f))
    except FileNotFoundError:
        hash_set = set()
    input_files = []

    logger.info(f"checking for duplicate docs....")
    for doc_path in raw_docs_path.glob("*.pdf"):
        with open(doc_path, "rb") as f:
            doc_hash = hashlib.sha256(f.read()).hexdigest()
        if doc_hash not in hash_set:
            hash_set.add(doc_hash)
            input_files.append(doc_path)
            

    logger.info(f"loading unique docs....")
    reader = SimpleDirectoryReader(
        input_files=input_files,
        file_extractor=file_extractor,
        recursive=True
    )
    
    logger.info(f"parsing docs....")
    documents = reader.load_data()
    logger.info(f"loaded {len(documents)} document pages....")

    # 2. Chunking
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=70,
        embed_model=Settings.embed_model
    )
    
    logger.info(f"chunking docs....")
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(f"created {len(nodes)} chunks....")

    # 3. Database + Index
    logger.info(f"connecting to Qdrant....")
    db_manager = VectorDBManager(
        db_path=db_path,
        collection_name=collection_name
    )      

    storage_context = db_manager.get_storage_context()

    logger.info(f"building index....")
    _ = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )

    logger.info(f"saving hashes....")
    with open(hash_data_path, "w") as f:
        json.dump(list(hash_set), f)

    logger.info(".......Ingestion Successful!!.......")


if __name__ == "__main__":
    ingestion()