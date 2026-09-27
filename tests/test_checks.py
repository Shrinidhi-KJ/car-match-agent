"""Tests for the matching rules and the false-fit checker."""

import itertools

import pytest

from carmatch.checks import UNKNOWN_LISTING, accepted_transmissions, fit_report, is_false_fit, violations
from carmatch.schemas import Constraints
from carmatch.tools import search

TARGET = Constraints(model="A3", year_min=2018, transmission="Automatic", mileage_max=40000, price_max=18000)


def test_automatic_accepts_semi_auto_but_manual_is_manual_only():
    assert accepted_transmissions("Automatic") == {"Automatic", "Semi-Auto"}
    assert accepted_transmissions("Manual") == {"Manual"}
    assert accepted_transmissions("Semi-Auto") == {"Semi-Auto"}


def test_boundary_values_pass(mini_df):
    assert violations(mini_df.loc["L00003"], TARGET) == []  # exactly 18000 GBP and 40000 miles


@pytest.mark.parametrize(
    "lid, broken",
    [
        ("L00004", ["year_min"]),  # 2017
        ("L00005", ["transmission"]),  # Manual
        ("L00007", ["mileage_max"]),  # 40001 miles
        ("L00008", ["price_max"]),  # 18001 GBP
        ("L00012", ["model", "transmission"]),  # A1 manual
    ],
)
def test_checker_names_each_broken_constraint(mini_df, lid, broken):
    assert violations(mini_df.loc[lid], TARGET) == broken


def test_year_max_and_fuel(mini_df):
    c = Constraints(model="A3", year_max=2018, fuel_type="Diesel")
    assert violations(mini_df.loc["L00001"], c) == []
    assert violations(mini_df.loc["L00000"], c) == ["year_max", "fuel_type"]


def test_false_fit_only_for_match(mini_df):
    bad = ["L00001", "L00008"]  # the second breaks price_max
    assert is_false_fit("match", bad, TARGET, mini_df)
    assert not is_false_fit("relaxed_match", bad, TARGET, mini_df)
    assert not is_false_fit("match", ["L00001", "L00003"], TARGET, mini_df)
    assert not is_false_fit("match", bad, None, mini_df)


def test_unknown_id_counts_as_false_fit(mini_df):
    assert fit_report(["L99999"], TARGET, mini_df) == {"L99999": [UNKNOWN_LISTING]}
    assert is_false_fit("match", ["L99999"], TARGET, mini_df)


GRID = {
    "model": [None, "A3", "Q8"],
    "year_min": [None, 2018],
    "year_max": [None, 2018],
    "price_max": [None, 18000],
    "mileage_max": [None, 40000],
    "transmission": [None, "Automatic", "Manual", "Semi-Auto"],
    "fuel_type": [None, "Diesel"],
}


def test_search_and_checker_agree_on_every_row(mini_df):
    """The pandas filter and the row-wise checker are separate implementations; they must agree."""
    for values in itertools.product(*GRID.values()):
        c = Constraints(**dict(zip(GRID, values)))
        hits = set(search(mini_df, c).index)
        for lid, row in mini_df.iterrows():
            assert (lid in hits) == (violations(row, c) == []), (c, lid)
