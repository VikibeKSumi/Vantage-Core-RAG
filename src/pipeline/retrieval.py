from typing import List
from llama_index.core.schema import NodeWithScore
from ..core.text_utils import TextUtils

class Retriever:

    def __init__(self):
        self.textutils = TextUtils()

    def retrieve(self, query: str, index, top_k: int = 20) -> List[NodeWithScore]:
        normalized_query = self.textutils.normalize(query=query)
        retriever = index.as_retriever(similarity_top_k=top_k) # <- LlamaIndex embeddes query internally
        return retriever.retrieve(normalized_query)