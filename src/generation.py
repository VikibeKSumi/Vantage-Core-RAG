from llama_index.llms.groq import Groq
from llama_index.core.schema import MetadataMode

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import RateLimitError, APIConnectionError


class Generator:
    """Handles response generation using Groq LLM."""

    def __init__(self, config):
        """Use centralized config (model + api_key)."""
        if not config.api_key:
            raise ValueError("Groq API Key is missing.")
        
        self.llm = Groq(
            model=config.models["llm"],
            api_key=config.api_key
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        reraise=True
    )

    def generate_response(self, query: str, context_nodes: list):
        """Returns answer + full token metrics (final version)."""
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

        # Extract token usage from Groq (LlamaIndex 2026 format)
        usage = getattr(response.raw, 'usage', None) if hasattr(response, 'raw') else None
        input_tokens = usage.prompt_tokens if usage else len(prompt.split())
        output_tokens = usage.completion_tokens if usage else len(response.text.split())

        # Reliable TPS calculation (fixed for LlamaIndex + Groq 2026)
        tps = output_tokens / getattr(response, 'response_time', 1)
        return {
            "answer": response.text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "tokens_per_second": round(tps, 2)
        }