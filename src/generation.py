import os
from llama_index.llms.groq import Groq
from llama_index.core.schema import MetadataMode


class Generator:
    def __init__(self,model: str, api_key: str):
        if not api_key:
            raise ValueError("Groq API Key is missing.")
        
        self.llm = Groq(model=model, api_key=api_key)

    def generate_response(self, query: str, context_nodes: list):
        context_text = "\n\n".join([
            node.node.get_content(metadata_mode=MetadataMode.NONE) 
            for node in context_nodes
        ])
        
        prompt = (
            f"Context Information:\n{context_text}\n\n"
            f"Query: {query}\n\n"
            "As an Advisor, provide a concise, grounded answer based strictly on the context. "
            "If the information is not present, admit it. Match the query's language."
        )
        
        response = self.llm.complete(prompt)
        return response.text