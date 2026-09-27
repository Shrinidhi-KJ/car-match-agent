"""Tests for eval/score.py on small invented examples (not gold data)."""

from eval.score import (
    constraints_match,
    first_search_args,
    score_task,
    summarize,
    summary_markdown,
    tools_called,
)

A3 = {"model": "A3", "year_min": 2018, "transmission": "Automatic", "mileage_max": 40000, "price_max": 18000}


def listing(lid, **kw):
    base = {"listing_id": lid, "model": "A3", "year": 2019, "price": 16000, "transmission": "Automatic",
            "mileage": 20000, "fuel_type": "Petrol", "engine_size": 1.5, "mpg": 47.1, "tax": 145}
    return base | kw


def step(name, args, status="success"):
    return {"name": name, "args": args, "status": status, "result": "{}"}


def task(**kw):
    base = {"id": "x1", "request": "r", "expected_constraints": A3, "expected_outcome": "match",
            "expected_relaxed": None, "expected_tools": ["search_listings", "submit_answer"], "broker_notes": ""}
    return base | kw


def run(**kw):
    base = {"outcome": "match", "listing_ids": ["L1"], "relaxed_constraint": None,
            "trace": [step("search_listings", A3), step("submit_answer", {})],
            "returned_listings": [listing("L1")], "llm_calls": 2, "input_tokens": 3000, "output_tokens": 200,
            "latency_s": 1.5, "error": None}
    return base | kw


def test_clean_match_all_correct():
    s = score_task(task(), run())
    assert s["outcome_correct"] and s["tools_correct"] and s["constraints_correct"]
    assert s["relaxed_correct"] is None
    assert s["false_fit"] is False


def test_tool_set_ignores_order_and_repeats():
    trace = [step("search_listings", A3), step("search_listings", A3), step("submit_answer", {})]
    assert tools_called(trace) == {"search_listings", "submit_answer"}
    assert score_task(task(), run(trace=trace))["tools_correct"]
    extra = trace + [step("market_summary", {"model": "A3"})]
    assert not score_task(task(), run(trace=extra))["tools_correct"]


def test_constraints_compared_after_normalisation_and_missing_equals_null():
    assert constraints_match({"model": "Audi a3", "year_min": 2018, "transmission": "automatic",
                              "mileage_max": 40000, "price_max": 18000, "fuel_type": None}, A3)
    assert not constraints_match(A3 | {"price_max": 17999}, A3)
    assert not constraints_match(A3 | {"fuel_type": "Petrol"}, A3)  # an extra constraint is wrong
    assert not constraints_match({"transmission": "CVT"}, A3)  # invalid args
    assert not constraints_match(None, A3)


def test_only_first_search_is_scored_for_parsing():
    trace = [step("search_listings", A3 | {"price_max": 15000}), step("search_listings", A3)]
    assert first_search_args(trace)["price_max"] == 15000
    assert score_task(task(), run(trace=trace))["constraints_correct"] is False


def test_constraints_metric_not_applicable_without_expected_search():
    t = task(expected_constraints=None, expected_outcome="ask_customer", expected_tools=["ask_customer"])
    s = score_task(t, run(outcome="ask_customer", listing_ids=[], returned_listings=[],
                          trace=[step("ask_customer", {"question": "?"})]))
    assert s["constraints_correct"] is None
    assert s["outcome_correct"] and s["tools_correct"]


def test_false_fit_judged_against_gold_constraints():
    over = listing("L2", price=18500)
    s = score_task(task(), run(listing_ids=["L1", "L2"], returned_listings=[listing("L1"), over]))
    assert s["false_fit"] is True
    assert s["false_fit_details"] == {"L2": ["price_max"]}


def test_false_fit_even_if_agent_searched_wrong_constraints():
    # Agent searched with a looser budget and called it a match: still a false fit against the gold.
    loose = A3 | {"price_max": 20000}
    s = score_task(task(), run(trace=[step("search_listings", loose), step("submit_answer", {})],
                              returned_listings=[listing("L1", price=19000)]))
    assert s["false_fit"] and not s["constraints_correct"]


def test_semi_auto_is_not_a_false_fit_for_automatic():
    s = score_task(task(), run(returned_listings=[listing("L1", transmission="Semi-Auto")]))
    assert s["false_fit"] is False


def test_unknown_listing_in_match_is_false_fit():
    assert score_task(task(), run(returned_listings=[None]))["false_fit"] is True


def test_relaxed_match_scoring():
    t = task(expected_outcome="relaxed_match", expected_relaxed="mileage_max",
             expected_tools=["search_listings", "market_summary", "submit_answer"])
    good = run(outcome="relaxed_match", relaxed_constraint="mileage_max",
               returned_listings=[listing("L1", mileage=45000)],
               trace=[step("search_listings", A3), step("market_summary", {"model": "A3"}), step("submit_answer", {})])
    s = score_task(t, good)
    assert s["relaxed_correct"] and s["outcome_correct"] and s["tools_correct"]
    assert s["false_fit"] is False  # false fit applies to "match" only
    assert s["relaxed_extra_violation"] is False

    wrong_name = score_task(t, good | {"relaxed_constraint": "price_max"})
    assert wrong_name["relaxed_correct"] is False

    also_over_budget = score_task(t, good | {"returned_listings": [listing("L1", mileage=45000, price=19000)]})
    assert also_over_budget["relaxed_extra_violation"] is True


def test_gave_up_and_error_are_always_wrong():
    s = score_task(task(), run(outcome="gave_up", listing_ids=[], returned_listings=[]))
    assert not s["outcome_correct"] and s["gave_up"]
    e = score_task(task(), run(outcome=None, error="boom", trace=[], listing_ids=[], returned_listings=[]))
    assert not e["outcome_correct"] and e["error"] == "boom"


def test_summary_counts_over_applicable_tasks_only():
    scores = [
        score_task(task(id="a"), run()),
        score_task(task(id="b"), run(outcome="gave_up", listing_ids=[], returned_listings=[])),
        score_task(task(id="c", expected_constraints=None, expected_outcome="ask_customer", expected_tools=["ask_customer"]),
                   run(outcome="ask_customer", listing_ids=[], returned_listings=[], trace=[step("ask_customer", {})])),
    ]
    summary = summarize(scores)
    assert summary["outcome_correct"] == {"correct": 2, "of": 3}
    assert summary["constraints_correct"] == {"correct": 2, "of": 2}
    assert summary["relaxed_correct"] == {"correct": 0, "of": 0}
    assert summary["gave_up"] == 1
    assert summary["input_tokens"] == 9000
    md = summary_markdown(summary, scores, "t")
    assert "| outcome correct | 2 of 3 |" in md
    assert "%" not in md
