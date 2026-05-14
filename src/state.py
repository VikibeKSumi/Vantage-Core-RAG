import torch
from typing import TypedDict, List, Optional
from llama_index.core.schema import NodeWithScore

class ResponseState(TypedDict):
    query : str
    rewritten_query: str
    cache_hit: Optional[bool]
    embedded_query: torch.Tensor
    retrieved_nodes: List[NodeWithScore]
    reranked_nodes: List[NodeWithScore]
    compressed_nodes: List[NodeWithScore]
    answer: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_per_second: int
