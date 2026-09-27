"""LangGraph build_graph(llm) -> compiled graph.

Nodes: agent (one LLM call), tools (runs the requested tools), record (reads the tool results),
nudge (reminds a model that replied without a tool call), gave_up (step cap hit).
Edges: START -> agent. agent -> tools if it made tool calls, else nudge (under the cap) or gave_up
(at the cap). tools -> record. record -> END if ask_customer or submit_answer succeeded, else
gave_up (at the cap) or agent. nudge -> agent. gave_up -> END.
"""

import hashlib
from collections.abc import Callable
from typing import Any

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from carmatch.checks import fit_report
from carmatch.data import load_listings
from carmatch.schemas import MAX_LLM_CALLS, AgentState, Constraints
from carmatch.tools import build_tools

# >>> PLACEHOLDER: Shrinidhi supplies the real relaxation policy (PLAN.md section 3). <<<
# The interim sentence only keeps the agent runnable until then; it is not the policy.
RELAXATION_POLICY = (
    "[PLACEHOLDER - relaxation policy not yet supplied.] "
    "Interim rule: loosen at most one constraint, and say in the message exactly which one and by how much."
)

SYSTEM_PROMPT_TEMPLATE = """You match a customer's used-car request to listings in a fixed dataset of UK used Audi listings (scraped around 2020).

How to work:
1. If the request is too vague to search (no model and no concrete requirement such as budget, year, mileage or gearbox), call ask_customer with one short question. Do not guess.
2. Otherwise call search_listings with exactly the constraints the customer stated. Do not add constraints they did not state.
3. If the search finds listings, call submit_answer with outcome "match" and up to 5 listing IDs in the order returned.
4. If the search finds nothing, call market_summary for that model to see which constraint is blocking. Then follow the relaxation policy below: loosen exactly one constraint and search again. If that finds listings, submit "relaxed_match", name the loosened constraint in relaxed_constraint, and say in the message by how much it was loosened. If nothing acceptable exists within the policy, submit "no_match" and say what does exist.

Rules:
- Never present a car that breaks any stated constraint as a "match".
- Never rank or reorder cars yourself; search_listings already ranks them.
- Only use listing IDs that appear in search_listings results.
- Always finish by calling submit_answer or ask_customer. You have at most {max_calls} turns.

Relaxation policy:
{policy}"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(max_calls=MAX_LLM_CALLS, policy=RELAXATION_POLICY)

NUDGE = (
    "[Note from the system, not the customer] You replied without calling a tool. "
    "Continue the task and finish by calling submit_answer or ask_customer."
)

LLMCaller = Callable[[Runnable, list[BaseMessage]], AIMessage]


def _default_caller(runnable: Runnable, messages: list[BaseMessage]) -> AIMessage:
    return runnable.invoke(messages)


def prompt_fingerprint(tools: list) -> str:
    """sha256 of the system prompt plus every tool's name, description and argument schema."""
    h = hashlib.sha256(SYSTEM_PROMPT.encode())
    for t in tools:
        h.update(f"\n{t.name}\n{t.description}\n{t.args}".encode())
    return h.hexdigest()


def _last_tool_round(messages: list[BaseMessage]) -> tuple[AIMessage | None, dict[str, ToolMessage]]:
    """The most recent AIMessage with tool calls, and its ToolMessages keyed by tool_call_id."""
    results: dict[str, ToolMessage] = {}
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            results[msg.tool_call_id] = msg
        elif isinstance(msg, AIMessage):
            return msg, results
    return None, results


def build_graph(
    llm: BaseChatModel,
    *,
    listings: pd.DataFrame | None = None,
    llm_caller: LLMCaller | None = None,
):
    """Compile the agent graph.

    listings: cleaned table from data.load_listings (default: the real Audi data).
    llm_caller: how each LLM call is made, e.g. the eval harness's paced, retrying, logging wrapper.
    """
    listings = load_listings() if listings is None else listings
    tools = build_tools(listings)
    bound = llm.bind_tools(tools)
    call = llm_caller or _default_caller

    def agent(state: AgentState) -> dict[str, Any]:
        reply = call(bound, [SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply], "llm_calls": state.get("llm_calls", 0) + 1}

    def record(state: AgentState) -> dict[str, Any]:
        """Read the tool round that just ran: remember the first search's constraints, and stop on a
        successful terminal tool. A terminal call that errored (bad arguments) does not end the run."""
        ai, results = _last_tool_round(state["messages"])
        update: dict[str, Any] = {}
        original = state.get("original_constraints")
        for tc in ai.tool_calls if ai else []:
            result = results.get(tc["id"])
            if result is None or result.status == "error":
                continue
            if tc["name"] == "search_listings" and original is None:
                original = Constraints(**tc["args"])
                update["original_constraints"] = original
            if tc["name"] == "ask_customer":
                update |= {"outcome": "ask_customer", "final_message": tc["args"].get("question"), "listing_ids": []}
                break
            if tc["name"] == "submit_answer":
                args = tc["args"]
                ids = list(args.get("listing_ids") or [])
                update |= {
                    "outcome": args["outcome"],
                    "listing_ids": ids,
                    "relaxed_constraint": args.get("relaxed_constraint"),
                    "final_message": args.get("message"),
                }
                if args["outcome"] == "match" and original is not None:
                    report = fit_report(ids, original, listings)
                    update["false_fit"] = any(report.values())
                    update["false_fit_details"] = {k: v for k, v in report.items() if v}
                break
        return update

    def nudge(state: AgentState) -> dict[str, Any]:
        return {"messages": [HumanMessage(NUDGE)]}

    def gave_up(state: AgentState) -> dict[str, Any]:
        return {"outcome": "gave_up"}

    def after_agent(state: AgentState) -> str:
        if state["messages"][-1].tool_calls:
            return "tools"
        return "gave_up" if state["llm_calls"] >= MAX_LLM_CALLS else "nudge"

    def after_record(state: AgentState) -> str:
        if state.get("outcome"):
            return END
        return "gave_up" if state["llm_calls"] >= MAX_LLM_CALLS else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("record", record)
    graph.add_node("nudge", nudge)
    graph.add_node("gave_up", gave_up)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "nudge", "gave_up"])
    graph.add_edge("tools", "record")
    graph.add_conditional_edges("record", after_record, ["agent", "gave_up", END])
    graph.add_edge("nudge", "agent")
    graph.add_edge("gave_up", END)
    compiled = graph.compile()
    compiled.prompt_fingerprint = prompt_fingerprint(tools)
    compiled.listings = listings
    return compiled


def initial_state(request: str) -> AgentState:
    return {"messages": [HumanMessage(request)], "llm_calls": 0}

