

import qdrant_client
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from ..config import Config 


class VectorDBManager:

    def __init__(self, config: Config):

        self.client = qdrant_client.QdrantClient(
            path=config['database']['path']
        )
        self.collection_name = config.data['database']['collection_name']

    def get_vector_store(self):
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            enable_hybrid=True
        )

    def get_storage_context(self):
        return StorageContext.from_defaults(vector_store=self.get_vector_store())