"""The four agent tools: search_listings, market_summary, ask_customer, submit_answer.

Tool descriptions below are what the LLM reads when choosing a tool, so they state exactly
when to use each one. Every tool returns a JSON string.
"""

import json
from typing import Any

import pandas as pd
from langchain_core.tools import BaseTool, StructuredTool, ToolException

from carmatch.checks import accepted_transmissions
from carmatch.schemas import (
    AskCustomerArgs,
    Constraints,
    MarketSummaryArgs,
    SearchListingsArgs,
    SubmitAnswerArgs,
)

MAX_RESULTS = 5
LISTING_FIELDS = ["model", "year", "price", "transmission", "mileage", "fuel_type", "engine_size", "mpg", "tax"]

SEARCH_DESCRIPTION = """Search the Audi used-car listings for cars that meet ALL the given constraints.
Call this first whenever the customer names a model or gives any concrete requirement.
Pass only constraints the customer actually stated; leave the rest out.
Limits are inclusive: "under £18,000" -> price_max=18000, "under 40k miles" -> mileage_max=40000,
"2018 or newer" -> year_min=2018. model is the bare code ("A3", not "Audi A3").
For any automatic gearbox request pass transmission="Automatic"; this also matches Semi-Auto listings.
Returns matching_count and up to 5 listings already ranked (cheapest first, then lowest mileage).
Do not re-rank them. If matching_count is 0, call market_summary to find which constraint is blocking."""

MARKET_DESCRIPTION = """Summarise what exists for one model in a year range: number of listings, price and
mileage min/median/max, and counts by transmission and fuel type. Use it after search_listings finds
nothing, to work out which single constraint is blocking and by how much. It returns no listings:
to offer cars, call search_listings again with only that one constraint loosened."""

ASK_DESCRIPTION = """Ask the customer one short clarifying question. This ENDS the conversation.
Use it only when the request is too vague to search (for example no model and no budget or other
concrete requirement). Do not use it after a search; finish with submit_answer instead."""

SUBMIT_DESCRIPTION = """Give the final answer. This ENDS the conversation.
outcome "match": every listing meets every constraint the customer stated.
outcome "relaxed_match": nothing meets everything, so exactly one constraint was loosened and the
listings come from a search where only that constraint changed; name it in relaxed_constraint.
outcome "no_match": nothing suitable, even with one constraint loosened; listing_ids must be empty.
listing_ids must be copied exactly from search_listings results, best first.
Never call a car a match if it breaks any stated constraint."""


def search(listings: pd.DataFrame, constraints: Constraints) -> pd.DataFrame:
    """Deterministic filter and rank: all constraints, inclusive limits; price asc, mileage asc, id asc."""
    c = constraints
    mask = pd.Series(True, index=listings.index)
    if c.model is not None:
        mask &= listings["model"] == c.model
    if c.year_min is not None:
        mask &= listings["year"] >= c.year_min
    if c.year_max is not None:
        mask &= listings["year"] <= c.year_max
    if c.price_max is not None:
        mask &= listings["price"] <= c.price_max
    if c.mileage_max is not None:
        mask &= listings["mileage"] <= c.mileage_max
    if c.transmission is not None:
        mask &= listings["transmission"].isin(accepted_transmissions(c.transmission))
    if c.fuel_type is not None:
        mask &= listings["fuel_type"] == c.fuel_type
    hits = listings[mask]
    return hits.assign(_id=hits.index).sort_values(["price", "mileage", "_id"]).drop(columns="_id")


def listing_record(listing_id: str, row: pd.Series) -> dict[str, Any]:
    record = {"listing_id": listing_id}
    for field in LISTING_FIELDS:
        value = row[field]
        record[field] = value.item() if hasattr(value, "item") else value
    return record


def _spread(series: pd.Series) -> dict[str, int] | None:
    if series.empty:
        return None
    return {"min": int(series.min()), "median": int(series.median()), "max": int(series.max())}


def build_tools(listings: pd.DataFrame) -> list[BaseTool]:
    """The four tools, bound to one cleaned listings table (from data.load_listings)."""
    known_models = sorted(listings["model"].unique())

    def search_listings(**kwargs: Any) -> str:
        constraints = Constraints(**kwargs)
        hits = search(listings, constraints)
        result: dict[str, Any] = {
            "constraints_applied": constraints.active(),
            "matching_count": len(hits),
            "listings": [listing_record(lid, row) for lid, row in hits.head(MAX_RESULTS).iterrows()],
        }
        if constraints.model is not None and constraints.model not in known_models:
            result["note"] = f"No model '{constraints.model}' in the data. Known models: {', '.join(known_models)}"
        return json.dumps(result)

    def market_summary(model: str, year_min: int | None = None, year_max: int | None = None) -> str:
        args = MarketSummaryArgs(model=model, year_min=year_min, year_max=year_max)
        rows = search(listings, Constraints(model=args.model, year_min=args.year_min, year_max=args.year_max))
        all_years = listings.loc[listings["model"] == args.model, "year"]
        result = {
            "model": args.model,
            "year_min": args.year_min,
            "year_max": args.year_max,
            "count": len(rows),
            "price": _spread(rows["price"]),
            "mileage": _spread(rows["mileage"]),
            "transmission_counts": {k: int(v) for k, v in rows["transmission"].value_counts().items()},
            "fuel_counts": {k: int(v) for k, v in rows["fuel_type"].value_counts().items()},
            "model_years_available": [int(all_years.min()), int(all_years.max())] if len(all_years) else None,
        }
        if args.model not in known_models:
            result["note"] = f"No model '{args.model}' in the data. Known models: {', '.join(known_models)}"
        return json.dumps(result)

    def ask_customer(question: str) -> str:
        return json.dumps({"status": "asked", "question": question})

    def submit_answer(**kwargs: Any) -> str:
        answer = SubmitAnswerArgs(**kwargs)
        unknown = [lid for lid in answer.listing_ids if lid not in listings.index]
        if unknown:
            raise ToolException(
                f"Unknown listing_ids {unknown}. Copy IDs exactly from search_listings results and submit again."
            )
        return json.dumps({"status": "submitted", "outcome": answer.outcome})

    specs = [
        (search_listings, "search_listings", SEARCH_DESCRIPTION, SearchListingsArgs),
        (market_summary, "market_summary", MARKET_DESCRIPTION, MarketSummaryArgs),
        (ask_customer, "ask_customer", ASK_DESCRIPTION, AskCustomerArgs),
        (submit_answer, "submit_answer", SUBMIT_DESCRIPTION, SubmitAnswerArgs),
    ]
    return [
        StructuredTool.from_function(
            func=func, name=name, description=desc, args_schema=schema, handle_tool_error=True
        )
        for func, name, desc, schema in specs
    ]
