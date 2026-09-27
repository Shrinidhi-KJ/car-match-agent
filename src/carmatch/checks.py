"""Matching rules and the false-fit checker.

The checker is deliberately written row by row in plain Python, separately from the pandas
filter in tools.search_listings, so the two implementations cross-check each other in tests.
Both follow the same rules (PLAN.md section 1): numeric limits are inclusive, and a
transmission of "Automatic" also accepts "Semi-Auto".
"""

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from carmatch.schemas import Constraints

UNKNOWN_LISTING = "unknown_listing"


def accepted_transmissions(requested: str) -> frozenset[str]:
    """Listing transmissions that satisfy a requested transmission."""
    if requested == "Automatic":
        return frozenset({"Automatic", "Semi-Auto"})
    return frozenset({requested})


def violations(listing: Mapping[str, Any], constraints: Constraints) -> list[str]:
    """Names of the constraints this listing breaks (empty list = it fits)."""
    c = constraints
    broken = []
    if c.model is not None and listing["model"] != c.model:
        broken.append("model")
    if c.year_min is not None and listing["year"] < c.year_min:
        broken.append("year_min")
    if c.year_max is not None and listing["year"] > c.year_max:
        broken.append("year_max")
    if c.price_max is not None and listing["price"] > c.price_max:
        broken.append("price_max")
    if c.mileage_max is not None and listing["mileage"] > c.mileage_max:
        broken.append("mileage_max")
    if c.transmission is not None and listing["transmission"] not in accepted_transmissions(c.transmission):
        broken.append("transmission")
    if c.fuel_type is not None and listing["fuel_type"] != c.fuel_type:
        broken.append("fuel_type")
    return broken


def fit_report(
    listing_ids: Iterable[str], constraints: Constraints, listings: pd.DataFrame
) -> dict[str, list[str]]:
    """For each returned ID, the constraints it breaks. IDs not in the data break 'unknown_listing'."""
    report = {}
    for lid in listing_ids:
        if lid not in listings.index:
            report[lid] = [UNKNOWN_LISTING]
        else:
            report[lid] = violations(listings.loc[lid], constraints)
    return report


def is_false_fit(
    outcome: str | None, listing_ids: Iterable[str], constraints: Constraints | None, listings: pd.DataFrame
) -> bool:
    """True when the agent claimed 'match' but some returned listing breaks an original constraint."""
    if outcome != "match" or constraints is None:
        return False
    return any(fit_report(listing_ids, constraints, listings).values())
