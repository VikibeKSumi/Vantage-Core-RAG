# src/database.py
from typing import Any, Dict, Union

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext

from .config import Config


class VectorDBManager:
    """Manages Qdrant vector database connection and storage context."""

    def __init__(self, config):
        """Accepts Config object or raw dict (for compatibility)."""

        self.config = config

        self.client = qdrant_client.QdrantClient(
            path=self.config['database']['path']
        )
    
    def get_vector_store(self):
        """Returns QdrantVectorStore with hybrid search enabled."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.config['database']['collection_name'],
            enable_hybrid=True
        )

    def get_storage_context(self):
        """Returns StorageContext for ingestion."""
        return StorageContext.from_defaults(vector_store=self.get_vector_store())