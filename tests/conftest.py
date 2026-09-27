"""Shared fixtures. Everything here uses tests/fixtures/mini.csv; nothing touches the network."""

from pathlib import Path

import pytest

from carmatch.data import load_listings
from carmatch.tools import build_tools

MINI_CSV = Path(__file__).parent / "fixtures" / "mini.csv"


@pytest.fixture(scope="session")
def mini_df():
    return load_listings(MINI_CSV)


@pytest.fixture(scope="session")
def tools(mini_df):
    return {t.name: t for t in build_tools(mini_df)}
