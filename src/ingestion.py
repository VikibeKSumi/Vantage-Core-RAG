from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class IngestionEngine:
    def __init__(self, model_name="BAAI/bge-m3"):
        # BGE-M3 is the powerhouse for Indic/Multilingual
        self.embed_model = HuggingFaceEmbedding(model_name=model_name)
        
    def process_document(self, raw_text: str):
        # 1. Normalize
        clean_text = self.processor.normalize_text(raw_text)
        
        # 2. Semantic Chunking
        # We use the embedding model itself to find where "meaning" changes
        parser = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
        )
        
        # Logic to return processed nodes for the Vector DB
        return parser.get_nodes_from_documents([clean_text])