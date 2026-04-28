import qdrant_client
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

class VectorDBManager:

    def __init__(self, db_path: str, collection_name: str):
        self.client = qdrant_client.QdrantClient(
            url="http://localhost:6333"
        )
        self.collection_name = collection_name

    def get_vector_store(self):
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            enable_hybrid=False
        )

    def get_storage_context(self):
        return StorageContext.from_defaults(
            vector_store=self.get_vector_store()
        )