"""Graph routing tests with a scripted fake chat model: deterministic, no network."""

import itertools

from langchain_core.messages import ToolMessage

from carmatch.graph import NUDGE, RELAXATION_POLICY, SYSTEM_PROMPT, build_graph, initial_state
from carmatch.schemas import MAX_LLM_CALLS, Constraints
from fakes import call, say, scripted

TARGET = {"model": "A3", "year_min": 2018, "transmission": "Automatic", "mileage_max": 40000, "price_max": 18000}


def run(mini_df, script, request="a request"):
    graph = build_graph(scripted(script), listings=mini_df)
    return graph.invoke(initial_state(request))


def tool_messages(state):
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def test_search_then_submit_ends_with_match(mini_df):
    state = run(mini_df, [
        call("search_listings", TARGET),
        call("submit_answer", {"outcome": "match", "listing_ids": ["L00001", "L00000"], "message": "Two fit."}),
    ])
    assert state["outcome"] == "match"
    assert state["listing_ids"] == ["L00001", "L00000"]
    assert state["llm_calls"] == 2
    assert state["false_fit"] is False
    assert state["original_constraints"] == Constraints(**TARGET)
    assert [m.name for m in tool_messages(state)] == ["search_listings", "submit_answer"]


def test_empty_search_then_market_summary_then_relaxed_submit(mini_df):
    tight = TARGET | {"price_max": 15000}
    state = run(mini_df, [
        call("search_listings", tight),
        call("market_summary", {"model": "A3", "year_min": 2018}),
        call("search_listings", TARGET),
        call("submit_answer", {
            "outcome": "relaxed_match", "listing_ids": ["L00001"], "relaxed_constraint": "price_max", "message": "£995 over."
        }),
    ])
    assert state["outcome"] == "relaxed_match"
    assert state["relaxed_constraint"] == "price_max"
    assert state["llm_calls"] == 4
    assert state["original_constraints"] == Constraints(**tight)  # the first search, not the relaxed one
    assert state.get("false_fit") is None  # only checked for "match"


def test_ask_customer_ends_immediately(mini_df):
    state = run(mini_df, [call("ask_customer", {"question": "Which model and budget?"})])
    assert state["outcome"] == "ask_customer"
    assert state["final_message"] == "Which model and budget?"
    assert state["llm_calls"] == 1
    assert state.get("original_constraints") is None


def test_never_terminal_is_stopped_at_cap_with_gave_up(mini_df):
    endless = (call("search_listings", {"model": "A3"}) for _ in itertools.count())
    state = run(mini_df, endless)
    assert state["outcome"] == "gave_up"
    assert state["llm_calls"] == MAX_LLM_CALLS
    assert len(tool_messages(state)) == MAX_LLM_CALLS


def test_text_only_replies_are_nudged_then_stopped_at_cap(mini_df):
    state = run(mini_df, (say("Let me think.") for _ in itertools.count()))
    assert state["outcome"] == "gave_up"
    assert state["llm_calls"] == MAX_LLM_CALLS
    assert sum(1 for m in state["messages"] if m.content == NUDGE) == MAX_LLM_CALLS - 1


def test_nudge_then_recover(mini_df):
    state = run(mini_df, [say("Hmm."), call("ask_customer", {"question": "Budget?"})])
    assert state["outcome"] == "ask_customer"
    assert state["llm_calls"] == 2


def test_terminal_on_last_allowed_call_still_counts(mini_df):
    script = [call("search_listings", {"model": "A3"}) for _ in range(MAX_LLM_CALLS - 1)]
    script.append(call("submit_answer", {"outcome": "no_match", "message": "None."}))
    state = run(mini_df, script)
    assert state["outcome"] == "no_match"
    assert state["llm_calls"] == MAX_LLM_CALLS


def test_match_with_non_fitting_listing_is_flagged_false_fit(mini_df):
    state = run(mini_df, [
        call("search_listings", TARGET),
        call("submit_answer", {"outcome": "match", "listing_ids": ["L00001", "L00008"], "message": "x"}),
    ])
    assert state["outcome"] == "match"
    assert state["false_fit"] is True
    assert state["false_fit_details"] == {"L00008": ["price_max"]}


def test_invalid_submit_does_not_end_run_and_agent_can_retry(mini_df):
    state = run(mini_df, [
        call("search_listings", TARGET),
        call("submit_answer", {"outcome": "relaxed_match", "listing_ids": ["L00001"], "message": "x"}),  # no relaxed_constraint
        call("submit_answer", {"outcome": "match", "listing_ids": ["L00001"], "message": "x"}),
    ])
    errors = [m for m in tool_messages(state) if m.status == "error"]
    assert len(errors) == 1 and errors[0].name == "submit_answer"
    assert state["outcome"] == "match"
    assert state["llm_calls"] == 3


def test_unknown_listing_id_is_rejected_then_retried(mini_df):
    state = run(mini_df, [
        call("search_listings", TARGET),
        call("submit_answer", {"outcome": "match", "listing_ids": ["L99999"], "message": "x"}),
        call("submit_answer", {"outcome": "match", "listing_ids": ["L00001"], "message": "x"}),
    ])
    assert "Unknown listing_ids" in tool_messages(state)[1].content
    assert state["listing_ids"] == ["L00001"]


def test_llm_caller_hook_wraps_every_call(mini_df):
    seen = []

    def caller(runnable, messages):
        seen.append(len(messages))
        return runnable.invoke(messages)

    graph = build_graph(
        scripted([call("search_listings", TARGET), call("ask_customer", {"question": "?"})]),
        listings=mini_df,
        llm_caller=caller,
    )
    graph.invoke(initial_state("r"))
    assert seen == [2, 4]  # system + request; then + AI tool call + tool result


def test_system_prompt_has_marked_placeholder_and_honesty_rule():
    assert "PLACEHOLDER" in RELAXATION_POLICY
    assert RELAXATION_POLICY in SYSTEM_PROMPT
    assert 'Never present a car that breaks any stated constraint as a "match"' in SYSTEM_PROMPT
