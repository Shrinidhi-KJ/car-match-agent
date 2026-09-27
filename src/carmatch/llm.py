"""Chat model factory (Groq primary, NVIDIA NIM fallback) configured from env."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

GROQ_MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "low"
NIM_MODEL = "meta/llama-3.3-70b-instruct"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_OUTPUT_TOKENS = 1024


def make_llm(provider: str | None = None, *, max_retries: int = 2) -> BaseChatModel:
    """Build the chat model. provider: "groq" (default) or "nim"; env CARMATCH_LLM overrides the default.

    max_retries is the client's own retry count; the eval harness passes 0 and does its own
    logged 429 backoff instead.
    """
    load_dotenv(ENV_PATH)
    provider = provider or os.getenv("CARMATCH_LLM", "groq")
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            reasoning_effort=REASONING_EFFORT,
            max_tokens=MAX_OUTPUT_TOKENS,
            max_retries=max_retries,
        )
    if provider == "nim":  # untested: no NVIDIA_API_KEY yet
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=NIM_MODEL,
            base_url=NIM_BASE_URL,
            api_key=os.environ["NVIDIA_API_KEY"],
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            max_retries=max_retries,
        )
    raise ValueError(f"unknown provider {provider!r}; expected 'groq' or 'nim'")
