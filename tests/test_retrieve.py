
from src.pipeline.retrieval import Retriever
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_index():
    
    mock_index = MagicMock()
    mock_index.as_retriever.return_value.retrieve.return_value = ["it's a query for test","query for testing"]
    return mock_index

@pytest.fixture
def retrieve_obj(): # <- no need to pass mocks. The mock is independent of the class.
    return Retriever()


def test_return(retrieve_obj, mock_index):
    # this is a test on a function
    result = retrieve_obj.retrieve("test query", index=mock_index, top_k=2)
    assert isinstance(result, list)


def test_top_k(retrieve_obj, mock_index):
    retrieve_obj.retrieve("test query", index=mock_index, top_k=2) # <- first run
    mock_index.as_retriever.assert_called_once_with(similarity_top_k=2) # <- and then let test check it


