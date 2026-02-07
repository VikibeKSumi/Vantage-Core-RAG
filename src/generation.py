import os
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate

class GenerationEngine:
    def __init__(self, llm):
        self.llm = llm

    def get_indic_prompt(self):
        """
        A custom prompt designed to keep the LLM grounded and 
        respecting the language of the query.
        """
        template = (
            "You are the RAG AI, a Language Architect. Use the provided context to answer the question.\n"
            "Context Information:\n"
            "----------------------\n"
            "{context_str}\n"
            "----------------------\n"
            "Instructions:\n"
            "1. Answer in the same language as the query.\n"
            "2. If the context doesn't contain the answer, say you don't know. Do not hallucinate.\n"
            "3. Cite the source (metadata) if available.\n"
            "Query: {query_str}\n"
            "Answer:"
        )
        return PromptTemplate(template)


class Generator:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API Key is missing.")
        
        # Using Llama 3.3 70B: High intelligence, low latency
        self.llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)

    def generate_response(self, query: str, context_nodes: list):
        context_text = "\n\n".join([node.text for node in context_nodes])
        
        prompt = (
            f"Context Information:\n{context_text}\n\n"
            f"Query: {query}\n\n"
            "As an Advisor, provide a concise, grounded answer based strictly on the context. "
            "If the information is not present, admit it. Match the query's language."
        )
        
        response = self.llm.complete(prompt)
        return response.text