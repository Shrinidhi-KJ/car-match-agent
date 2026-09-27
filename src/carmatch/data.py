"""Load and clean the Audi CSV and assign stable listing IDs."""

import logging
from pathlib import Path

import pandas as pd

from carmatch.schemas import FUEL_TYPES, TRANSMISSIONS

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "audi.csv"

RAW_COLUMNS = ["model", "year", "price", "transmission", "mileage", "fuelType", "tax", "mpg", "engineSize"]
RENAMES = {"fuelType": "fuel_type", "engineSize": "engine_size"}


def listing_id(row_number: int) -> str:
    """ID from the row's position in the raw file (0-based, header excluded): row 0 -> 'L00000'."""
    return f"L{row_number:05d}"


def load_listings(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the CSV, strip whitespace, drop exact duplicate rows, and index by stable listing_id.

    IDs come from each row's position in the raw file before de-duplication, so dropping a
    duplicate never shifts any other listing's ID. The first copy of a duplicate is kept.
    """
    raw = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df = raw[RAW_COLUMNS].rename(columns=RENAMES)
    for col in ("model", "transmission", "fuel_type"):
        df[col] = df[col].astype(str).str.strip()

    bad_trans = set(df["transmission"]) - set(TRANSMISSIONS)
    bad_fuel = set(df["fuel_type"]) - set(FUEL_TYPES)
    if bad_trans or bad_fuel:
        raise ValueError(f"{path}: unexpected transmission {bad_trans or '-'} / fuel type {bad_fuel or '-'}")

    df.index = pd.Index([listing_id(i) for i in range(len(df))], name="listing_id")
    before = len(df)
    df = df[~df.duplicated(keep="first")]
    dropped = before - len(df)
    logger.info("loaded %s: %d rows, dropped %d exact duplicates, %d listings", path, before, dropped, len(df))
    df.attrs["duplicates_dropped"] = dropped
    return df
