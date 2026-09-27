"""One real agent run against Groq and the real Audi data. Marked slow: excluded from CI.

The request is written here, not taken from eval/gold/. It checks the pipeline end to end
(real tool calling, real tools, graph ends cleanly), not answer quality.
"""

import os

import pytest

from carmatch.llm import ENV_PATH, make_llm
from dotenv import load_dotenv

load_dotenv(ENV_PATH)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"),
]

REQUEST = "Looking for a 2019 or newer Audi Q3, automatic, under 30,000 miles, budget £25,000."


def test_one_real_run_end_to_end():
    from carmatch.graph import build_graph, initial_state
    from carmatch.schemas import MAX_LLM_CALLS
    from eval.run_eval import build_trace

    graph = build_graph(make_llm())
    state = graph.invoke(initial_state(REQUEST))
    trace = build_trace(state["messages"])

    # ascii() because the Windows console can't encode every character the model writes.
    print(ascii({"outcome": state.get("outcome"), "calls": state["llm_calls"], "ids": state.get("listing_ids")}))
    for step in trace:
        print(ascii((step["name"], step["args"], step["status"])))

    assert 1 <= state["llm_calls"] <= MAX_LLM_CALLS
    assert state["outcome"] in {"match", "relaxed_match", "no_match", "ask_customer", "gave_up"}
    searches = [s for s in trace if s["name"] == "search_listings"]
    assert searches, "the agent never searched"
    assert state["original_constraints"].model == "Q3"
