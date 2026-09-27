"""Tests for data loading and cleaning."""

import pandas as pd
import pytest

from carmatch.data import listing_id, load_listings
from conftest import MINI_CSV

HEADER = "model,year,price,transmission,mileage,fuelType,tax,mpg,engineSize\n"


def test_row_count_after_dropping_the_one_duplicate(mini_df):
    assert len(pd.read_csv(MINI_CSV)) == 32
    assert len(mini_df) == 31
    assert mini_df.attrs["duplicates_dropped"] == 1


def test_duplicate_keeps_first_copy_and_other_ids_do_not_shift(mini_df):
    assert "L00000" in mini_df.index
    assert "L00010" not in mini_df.index  # exact copy of row 0
    assert mini_df.loc["L00011", "price"] == 22995  # the row after the dropped one keeps its own ID


def test_ids_are_stable_across_loads(mini_df):
    pd.testing.assert_frame_equal(mini_df, load_listings(MINI_CSV))


def test_whitespace_stripped_and_columns_renamed(mini_df):
    assert {"A3", "Q8", "TT"} <= set(mini_df["model"])
    assert not mini_df["model"].str.contains(" ").any()
    assert {"fuel_type", "engine_size"} <= set(mini_df.columns)
    assert "fuelType" not in mini_df.columns


def test_listing_id_format():
    assert listing_id(0) == "L00000"
    assert listing_id(10667) == "L10667"


def test_unknown_transmission_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(HEADER + " A3,2019,1,CVT,1,Petrol,1,1.0,1.0\n")
    with pytest.raises(ValueError, match="transmission"):
        load_listings(bad)


def test_missing_column_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("model,year\n A3,2019\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_listings(bad)
