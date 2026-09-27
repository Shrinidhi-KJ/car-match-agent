"""Phase 0 smoke test: is the Groq model available, and does it return a tool call?

Usage: python scripts/smoke_llm.py [--nim] [--model MODEL_ID]
"""

import os
import sys

from dotenv import load_dotenv
from langchain_core.tools import tool

GROQ_MODEL = "llama-3.3-70b-versatile"
NIM_MODEL = "meta/llama-3.3-70b-instruct"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


@tool
def search_listings(model: str, price_max: int) -> str:
    """Search used Audi listings by model and maximum price in GBP."""
    return "[]"


def groq_llm(model: str):
    from groq import Groq
    from langchain_groq import ChatGroq

    available = sorted(m.id for m in Groq().models.list().data)
    print(f"{model} available on Groq: {model in available}")
    if model not in available:
        print("Available models:", ", ".join(available))
    return ChatGroq(model=model, temperature=0)


def nim_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=NIM_MODEL, base_url=NIM_BASE_URL, api_key=os.environ["NVIDIA_API_KEY"], temperature=0
    )


def main() -> int:
    load_dotenv()
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else GROQ_MODEL
    llm = nim_llm() if "--nim" in sys.argv else groq_llm(model)
    reply = llm.bind_tools([search_listings]).invoke(
        "Find me an Audi A3 under 18000 pounds. Use the search tool."
    )
    print("tool_calls:", reply.tool_calls)
    print("usage:", reply.usage_metadata)
    return 0 if reply.tool_calls else 1


if __name__ == "__main__":
    sys.exit(main())
