

from src.pipeline.reranker import Reranker
from llama_index.core.schema import NodeWithScore, TextNode
from unittest.mock import MagicMock
import pytest


@pytest.fixture
def mock_rerank_model():
    mock_model = MagicMock()
    mock_model.postprocess_nodes.return_value = []
    return mock_model

@pytest.fixture
def rerank(mock_rerank_model): # <- mocks passed here to be used
    return Reranker(reranking_model=mock_rerank_model) # <- mocks used here and becomes connected with the real one.

def test_rerank(rerank):
    reranked_response = rerank.rerank(query="test query", retrieved_response=[NodeWithScore(node=TextNode(text="test"), score=1.0)])
    assert isinstance(reranked_response, list)

def test_args(rerank, mock_rerank_model):
    nodes = [NodeWithScore(node=TextNode(text="test"), score=1.0)]
    rerank.rerank(query="test query", retrieved_response=nodes) # <- The mock records everything that happens to it, then interrogate it after.
    mock_rerank_model.postprocess_nodes.assert_called_once_with(nodes, query_str="test query")

