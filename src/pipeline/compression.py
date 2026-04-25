from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.schema import NodeWithScore
from typing import List

class ContextCompressor:
    """Compresses retrieved context to reduce tokens sent to Groq."""

    def __init__(self):
        self.reorder = LongContextReorder()

    def compress(self, reranked_nodes: list[NodeWithScore], max_tokens: int = 1800) -> List[NodeWithScore]:
 
        # Reorder for better LLM attention
        reordered = self.reorder.postprocess_nodes(reranked_nodes)
        
        # Simple token-based trimming (we can make it smarter later)
        compressed = []
        total_tokens = 0

        for node in reordered:
            node_tokens = len(node.node.get_content().split())
            if total_tokens + node_tokens > max_tokens:
                break
            compressed.append(node)
            total_tokens += node_tokens

        return compressed