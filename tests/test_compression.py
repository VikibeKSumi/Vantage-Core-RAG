

from src.pipeline.compression import ContextCompressor
import pytest
from llama_index.core.schema import NodeWithScore, TextNode


@pytest.fixture
def compress_obj():
    return ContextCompressor()

def test_empty(compress_obj):
    retrieved_nodes = []
    result = compress_obj.compress(retrieved_nodes)
    assert len(result) == 0

def test_max_token(compress_obj):
    retrieved_nodes= [
        NodeWithScore(node=TextNode(text="one two three four five"),  score=0.2),
        NodeWithScore(node=TextNode(text="six seven eight nine ten"),  score=0.2),
        NodeWithScore(node=TextNode(text="eleven twelve thirteen fourteen fifteen"),  score=0.2)
    ]
    result = compress_obj.compress(retrieved_nodes, max_tokens=8)
    assert len(result) <= 8