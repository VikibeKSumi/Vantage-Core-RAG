import yaml
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.database import VectorDBManager

def main():
    #Setup Configuration
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    
    # 1. Load Document
    print("--- Ingesting Industrial Sources ---")
    reader = SimpleDirectoryReader(
        input_dir="./data/raw_docs", 
        recursive=True
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document pages.")

    #2. Parsing
    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(documents)

    
    # 3. Turn chunked nodes into stored vectors
    #Initialize Database
    db_manager = VectorDBManager(config)
    storage_context = db_manager.get_storage_context()
    
    print("--- Initializing Embedding Model ---")
    llama_embed_model = HuggingFaceEmbedding(
        model_name=config['models']['embedding'], 
        device="cpu"
    )

    _ = VectorStoreIndex(
        nodes, 
        storage_context=storage_context,
        embed_model=llama_embed_model 
    )
    
    print("✅ Industrial Ingestion Complete.")


if __name__ == "__main__":
    main()