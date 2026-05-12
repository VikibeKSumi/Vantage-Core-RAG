import sys
from pathlib import Path
import json
import hashlib

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from loguru import logger
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import Document
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter  # swap to SemanticSplitterNodeParser for better quality (slow on CPU)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.config import config
from src.services.vector_store import VectorDBManager


def unique_files(hash_data_path: Path, raw_docs_path: Path):

    hash_data_path.parent.mkdir(parents=True, exist_ok=True)

    try: 
        with open(hash_data_path, "r") as f:
            hash_set = set(json.load(f))
    except FileNotFoundError:
        hash_set = set()
    input_files = []
    
    logger.info(f"checking for duplicate docs and filtereing for unique files........")
    for doc_path in raw_docs_path.glob("*.pdf"):
        with open(doc_path, "rb") as f:
            doc_hash = hashlib.sha256(f.read()).hexdigest()
        if doc_hash not in hash_set: #<- checks here
            hash_set.add(doc_hash) #<- unique hash here
            input_files.append(doc_path) #<- unique files here

    return input_files, hash_set


def add_metadata(metadata_path: Path, documents: Document):

    with open(metadata_path, "r") as f:
        metadatas: dict = json.load(f)
    
    logger.info(f"adding metadata....")
    for doc in documents:
        for key in metadatas:
            if doc.metadata.get("file_name").startswith(key):
                doc.metadata.update(metadatas.get(key))
    
    return documents


def ingestion() -> None:

    logger.info(f".........Starting Ingestion Pipeline.........")
    
    db_path = str(Path(config.database.get("db_path")))
    raw_docs_path = Path(config.database.get("raw_data_path"))
    hash_data_path = Path("data/hash_store") / "hash_data.json"
    metadata_path = Path("data/raw_docs") / "_metadatas.json"
    collection_name = config.database.get("collection_name")
    embed_model_name = config.models.get("embedding")
    device = "cuda" if torch.cuda.is_available() else "cpu" #<- utilizing cpu due to GPU overheat

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        device=device
    )

    parser = PyMuPDFReader()
    file_extractor = {'.pdf': parser}

    input_files, hash_set = unique_files(hash_data_path, raw_docs_path)    

    logger.info(f"loading unique docs....")
    reader = SimpleDirectoryReader(
        input_files=input_files,
        file_extractor=file_extractor,
        recursive=True
    )

    logger.info(f"parsing docs....")
    documents= reader.load_data()
    logger.info(f"loaded {len(documents)} document pages....")

    documents = add_metadata(metadata_path, documents)

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    ) # <- Normal splitter implemented due to resource bottleneck in semantic splitter
    
    logger.info(f"chunking docs....")
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(f"created {len(nodes)} chunks....")

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