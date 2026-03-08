# src/compressor.py — Context Compression (production-ready)
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.schema import NodeWithScore


class ContextCompressor:
    """Compresses retrieved context to reduce tokens sent to Groq."""

    def __init__(self):
        self.reorder = LongContextReorder()

    def compress(self, nodes: list[NodeWithScore], max_tokens: int = 1800) -> list[NodeWithScore]:
        """
        1. Reorders nodes so most relevant are in the middle (LLM reads better)
        2. Trims to target token count
        """
        # Reorder for better LLM attention
        reordered = self.reorder.postprocess_nodes(nodes)
        
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