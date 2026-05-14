from typing import List
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex
from ..core.text_utils import TextUtils
from ..state import ResponseState


class Retriever:

    def __init__(self, index: VectorStoreIndex, top_k: int = 20):
        self.textutils = TextUtils()
        self.index = index
        self.top_k = top_k


    def retrieve(self, state: ResponseState) -> List[NodeWithScore]:
        rewritten_query = state.get("rewritten_query")
        normalized_query = self.textutils.normalize(query=rewritten_query)
        retriever = self.index.as_retriever(similarity_top_k=self.top_k) # <- LlamaIndex embeddes query internally

        return {"retrieved_nodes": retriever.retrieve(normalized_query)}