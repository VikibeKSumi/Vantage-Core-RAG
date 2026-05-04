

from src.pipeline.cache import SemanticCache
from unittest.mock import MagicMock
import pytest
import torch



@pytest.fixture
def cache():
    mock_model = MagicMock()
    mock_model.get_text_embedding.return_value = [1.0, 0.0, 0.0]
    return SemanticCache(embedding_model=mock_model, cache_similarity_threshold=0.92)

def test_empty_cache(cache):
    result, _ = cache.get("test query")
    assert result is None

def test_miss_cache(cache):
    cache.store(query="test query", result={}, embedded_query=torch.tensor([0.0, 1.0, 0.0]))
    result, _ = cache.get("test query")
    assert result is False


def test_hit_cache(cache):
    cache.store(query="test query", result={}, embedded_query=torch.tensor([1.0, 0.0, 0.0]))
    result, _ = cache.get("test query")
    assert result is True
