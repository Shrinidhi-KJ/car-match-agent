"""Tests for the four tools, against tests/fixtures/mini.csv."""

import json

import pytest
from pydantic import ValidationError

from carmatch.schemas import Constraints
from carmatch.tools import search

TARGET = {"model": "A3", "year_min": 2018, "transmission": "Automatic", "mileage_max": 40000, "price_max": 18000}


def run(tools, name, args):
    return json.loads(tools[name].invoke(args))


def ids(result):
    return [x["listing_id"] for x in result["listings"]]


def test_target_query_ranked_by_price_then_mileage(tools):
    r = run(tools, "search_listings", TARGET)
    assert r["matching_count"] == 5
    # L00001 and L00000 tie on price; lower mileage first. L00006 is Semi-Auto (counts as automatic).
    assert ids(r) == ["L00001", "L00000", "L00006", "L00002", "L00003"]
    assert r["constraints_applied"] == TARGET


def test_returns_at_most_five_cheapest(tools, mini_df):
    r = run(tools, "search_listings", {})
    assert r["matching_count"] == 31
    assert [x["price"] for x in r["listings"]] == sorted(mini_df["price"])[:5]


def test_empty_result(tools):
    r = run(tools, "search_listings", {"model": "Q8", "price_max": 20000})
    assert r["matching_count"] == 0
    assert r["listings"] == []


@pytest.mark.parametrize(
    "constraint, near_miss",
    [
        ({"year_min": 2018}, "L00004"),
        ({"price_max": 18000}, "L00008"),
        ({"mileage_max": 40000}, "L00007"),
        ({"transmission": "Automatic"}, "L00005"),
    ],
)
def test_each_constraint_filters_its_near_miss(tools, mini_df, constraint, near_miss):
    unfiltered = run(tools, "search_listings", {"model": "A3"})
    filtered = run(tools, "search_listings", {"model": "A3"} | constraint)
    assert filtered["matching_count"] < unfiltered["matching_count"]
    # The tool shows only 5, so check the full ranked hit set.
    assert near_miss in search(mini_df, Constraints(model="A3")).index
    assert near_miss not in search(mini_df, Constraints(model="A3", **constraint)).index


def test_manual_excludes_semi_auto_and_semi_auto_is_exact(tools):
    assert ids(run(tools, "search_listings", {"model": "A3", "transmission": "Manual"})) == ["L00009", "L00005"]
    assert ids(run(tools, "search_listings", {"model": "A3", "transmission": "Semi-Auto"})) == ["L00006"]


def test_model_and_enum_normalisation(tools):
    r = run(tools, "search_listings", TARGET | {"model": "Audi a3", "transmission": "automatic"})
    assert r["constraints_applied"] == TARGET
    assert r["matching_count"] == 5


def test_unknown_model_gets_a_note(tools):
    r = run(tools, "search_listings", {"model": "A9"})
    assert r["matching_count"] == 0
    assert "Known models" in r["note"]


def test_invalid_arguments_rejected(tools):
    with pytest.raises(ValidationError):
        tools["search_listings"].invoke({"transmission": "CVT"})
    with pytest.raises(ValidationError):
        Constraints(colour="red")


def test_market_summary_counts_and_spread(tools):
    r = run(tools, "market_summary", {"model": "A3", "year_min": 2018, "year_max": 2019})
    assert r["count"] == 8  # L00000, 01, 03, 05, 06, 07, 08, 11
    assert r["price"] == {"min": 15200, "median": 16450, "max": 22995}
    assert r["transmission_counts"] == {"Automatic": 6, "Manual": 1, "Semi-Auto": 1}


def test_market_summary_empty_range(tools):
    r = run(tools, "market_summary", {"model": "Q8", "year_min": 2015, "year_max": 2018})
    assert r["count"] == 0
    assert r["price"] is None and r["mileage"] is None
    assert r["model_years_available"] == [2019, 2020]


def test_ask_customer_echoes_question(tools):
    assert run(tools, "ask_customer", {"question": "Which model?"}) == {"status": "asked", "question": "Which model?"}


def test_submit_answer_accepts_valid(tools):
    r = run(tools, "submit_answer", {"outcome": "match", "listing_ids": ["L00001"], "message": "ok"})
    assert r == {"status": "submitted", "outcome": "match"}


def test_submit_answer_unknown_id_is_a_tool_error_not_a_crash(tools):
    out = tools["submit_answer"].invoke({"outcome": "match", "listing_ids": ["L99999"], "message": "x"})
    assert "Unknown listing_ids" in out


@pytest.mark.parametrize(
    "args",
    [
        {"outcome": "relaxed_match", "listing_ids": ["L00001"], "message": "x"},  # no relaxed_constraint
        {"outcome": "match", "listing_ids": ["L00001"], "relaxed_constraint": "price_max", "message": "x"},
        {"outcome": "match", "listing_ids": [], "message": "x"},
        {"outcome": "no_match", "listing_ids": ["L00001"], "message": "x"},
        {"outcome": "gave_up", "listing_ids": [], "message": "x"},  # the agent may not submit gave_up
    ],
)
def test_submit_answer_inconsistent_args_rejected(tools, args):
    with pytest.raises(ValidationError):
        tools["submit_answer"].invoke(args)
