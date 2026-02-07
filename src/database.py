import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext

class VectorDBManager:
    def __init__(self, config):
        self.config = config
        # Initialize local Qdrant client
        self.client = qdrant_client.QdrantClient(
            path=self.config['database']['path']
        )
        
    def get_vector_store(self):
        """
        Creates a QdrantVectorStore instance. 
        Qdrant supports hybrid search which we will use later.
        """
        return QdrantVectorStore(
            client=self.client, 
            collection_name=self.config['database']['collection_name'],
            enable_hybrid=True # Crucial for Indic keyword + semantic search
        )

    def get_storage_context(self):
        vector_store = self.get_vector_store()
        return StorageContext.from_defaults(vector_store=vector_store)