"""Shared data models (PLAN.md sections 1 and 3).

Every lane imports from here, so field names in this file are the contract between
data/tools (Lane A), the graph (Lane B), scoring (Lane C) and the hand-written gold set.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

Transmission = Literal["Manual", "Automatic", "Semi-Auto"]
FuelType = Literal["Petrol", "Diesel", "Hybrid"]

# What the agent may submit via submit_answer.
SubmitOutcome = Literal["match", "relaxed_match", "no_match"]
# How a run ended. "ask_customer" = run ended with a clarifying question;
# "gave_up" = step cap hit without a terminal tool (always scored wrong).
RunOutcome = Literal["match", "relaxed_match", "no_match", "ask_customer", "gave_up"]

ToolName = Literal["search_listings", "market_summary", "ask_customer", "submit_answer"]
TERMINAL_TOOLS: frozenset[str] = frozenset({"ask_customer", "submit_answer"})
MAX_LLM_CALLS = 6


class Constraints(BaseModel):
    """A customer's hard requirements. Every bound is inclusive; None means no constraint."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(None, description="Audi model code without spaces, e.g. 'A3', 'Q5', 'RS3'.")
    year_min: int | None = Field(None, description="Earliest registration year, inclusive.")
    year_max: int | None = Field(None, description="Latest registration year, inclusive.")
    price_max: int | None = Field(None, description="Maximum price in GBP, inclusive.")
    mileage_max: int | None = Field(None, description="Maximum mileage in miles, inclusive.")
    transmission: Transmission | None = None
    fuel_type: FuelType | None = None


ConstraintName = Literal[
    "model", "year_min", "year_max", "price_max", "mileage_max", "transmission", "fuel_type"
]


class Listing(BaseModel):
    """One row of the cleaned dataset."""

    listing_id: str
    model: str
    year: int
    price: int
    transmission: Transmission
    mileage: int
    fuel_type: FuelType
    tax: int
    mpg: float
    engine_size: float


# ---- Tool argument schemas (Lane A's tools and Lane B's stubs both use these) ----


class SearchListingsArgs(BaseModel):
    constraints: Constraints


class MarketSummaryArgs(BaseModel):
    model: str
    year_min: int | None = None
    year_max: int | None = None


class AskCustomerArgs(BaseModel):
    question: str


class SubmitAnswerArgs(BaseModel):
    outcome: SubmitOutcome
    listing_ids: list[str] = Field(default_factory=list, max_length=5)
    relaxed_constraint: ConstraintName | None = None
    message: str


# ---- Graph state ----


class AgentState(TypedDict, total=False):
    """LangGraph state. A TypedDict (not Pydantic) so the add_messages reducer works as documented."""

    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int
    original_constraints: Constraints | None  # args of the first search_listings call
    outcome: RunOutcome | None
    listing_ids: list[str]
    relaxed_constraint: ConstraintName | None
    final_message: str | None  # submit_answer message or ask_customer question


# ---- Gold set (PLAN.md section 3) ----


class GoldTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    request: str
    expected_constraints: Constraints | None
    expected_outcome: RunOutcome
    expected_relaxed: ConstraintName | None
    expected_tools: list[ToolName]
    broker_notes: str
