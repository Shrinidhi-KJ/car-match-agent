# Decisions log

Format: decision, reason, alternative rejected. Newest at the bottom of each section.

---

## Phase 0 (2026-09-27)

### Environment and pinned versions

Python 3.11.5, venv at `.venv/`, package installed editable (`pip install -e ".[dev]"`). Versions are pinned exactly in `pyproject.toml`.

| package | version |
|---|---|
| langgraph | 1.2.12 (langgraph-prebuilt 1.1.0, langgraph-checkpoint 4.2.0) |
| langchain-core | 1.6.5 |
| langchain-groq | 1.1.3 (groq 0.37.1) |
| langchain-openai | 1.6.6 (openai 3.19.2) |
| pandas | 3.0.6 |
| pydantic | 2.13.5 |
| python-dotenv | 1.2.3 |
| pytest | 9.1.1 |

### LangGraph API notes (checked against the installed package, not docs from memory)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
```

- `StateGraph.add_conditional_edges(source, path, path_map=None)`: `path` is a callable returning a node name (or `END`); `path_map` maps return values to node names.
- `ToolNode(tools, *, name="tools", handle_tool_errors=..., messages_key="messages")`: the default `handle_tool_errors` turns tool exceptions into ToolMessages instead of raising.
- `tools_condition(state) -> "tools" | "__end__"`: too simple for us (no terminal-tool or step-cap routing). Lane B should write its own router.
- `compile(checkpointer=None, ..., debug=False)`. `invoke(..., config={"recursion_limit": n})` raises `GraphRecursionError`. Don't rely on that for the 6-call cap; count `llm_calls` in state and route to a gave_up node instead.
- **Fake model:** `GenericFakeChatModel(messages=iter([...AIMessage...]))` replays scripted AIMessages (with `tool_calls`), but **`bind_tools` raises `NotImplementedError`**. Tests need a tiny subclass with `def bind_tools(self, tools, **kw): return self`.

### Data findings (`data/raw/audi.csv`)

- Source: Kaggle "100,000 UK Used Car Data set" by adityadesai13, file `audi.csv`. **Licence: CC0: Public Domain** (read on the dataset page on 2026-09-27). Redistribution is allowed.
- It downloaded from `https://www.kaggle.com/api/v1/datasets/download/adityadesai13/used-car-dataset-ford-and-mercedes/audi.csv` **without a login** (PLAN.md says Kaggle needs one). sha256 `f41fcda6795e500dbba178e44eefd247cfdcd31e2a564fbea2f8baa327167481`.
- 10,668 rows, 9 columns, no nulls: `model` (str), `year` (int), `price` (int, GBP), `transmission` (str), `mileage` (int), `fuelType` (str), `tax` (int), `mpg` (float), `engineSize` (float).
- **Every `model` value has a leading space** (`" A3"`). There are 26 models. Counts: A3 1929, Q3 1417, A4 1381, A1 1347, A5 882, Q5 877, Q2 822, A6 748, Q7 397, TT 336, A7 122, A8 118, Q8 69, RS6 39, RS3 33, RS4 31, RS5 29, R8 28, S3 18, SQ5 16, S4 12, SQ7 8, S8 4, S5 3, A2 1, RS7 1. `transmission` and `fuelType` have no stray whitespace.
- **`transmission` has three values: Manual 4369, Semi-Auto 3591, Automatic 2708.** PLAN.md assumed Manual/Automatic. Whether a customer's "automatic" includes Semi-Auto is a broker decision (see open questions).
- `fuelType`: Diesel 5577, Petrol 5063, Hybrid 28.
- Ranges (min / median / max): year 1997 / 2017 / 2020; price £1,490 / £20,200 / £145,000; mileage 1 / 19,000 / 323,000. Most listings are 2015 to 2020, with only 2 before 2003.
- Dirty or odd values: **103 exact duplicate rows**; 57 rows with `engineSize` 0.0; 90 rows with mileage ≤ 10 (likely nearly-new or pre-registered cars). No negative or zero prices.
- The PLAN.md example request (A3, 2018 or newer, Automatic, ≤40,000 mi, ≤£18,000, with Automatic strictly excluding Semi-Auto) matches 3 listings.

### Schema decisions (`src/carmatch/schemas.py`)

- **Constraint bounds are inclusive** (`mileage_max=40000` keeps a 40,000-mile car). Reason: one rule for all bounds and it's easy to check. Rejected: strict "under", which would make the gold author write 39999. Gold `expected_constraints` should write "under 40k" as `40000`.
- **Constraint fields:** `model, year_min, year_max, price_max, mileage_max, transmission, fuel_type`, with `extra="forbid"`. `model` is the code without spaces (`"A3"`, not `"Audi A3"` or `" A3"`). Rejected: price_min, engine size, mpg and tax as constraints, because customers rarely state them and each one adds parsing surface.
- **Run outcomes:** `match | relaxed_match | no_match | ask_customer | gave_up`. `ask_customer` is the outcome of a run that ended with a clarifying question. PLAN.md didn't name one, and the "too vague" gold tasks need an `expected_outcome`. `submit_answer` only accepts the first three, so the model can never submit `gave_up`.
- **`relaxed_constraint` is a Literal of the constraint field names**, so scoring compares exact strings.
- **`AgentState` is a TypedDict, not a Pydantic model**, because it's the documented way to use the `add_messages` reducer. It carries `llm_calls`, `original_constraints` (the first `search_listings` args, for the false-fit check), the outcome and the final fields.
- **Tool argument schemas** (`SearchListingsArgs` etc.) live in schemas.py so Lane A's tools and Lane B's stubs share one signature.
- **`GoldTask` model** validates gold lines. `expected_constraints` is nullable for tasks where no search is expected.

### Fixture (`tests/fixtures/mini.csv`, 32 invented rows, same columns and the leading space on model)

Built around the query A3 / year_min 2018 / Automatic / mileage_max 40000 / price_max 18000:
- 4 matches, including a price tie (£15,995) broken by mileage, and exact-boundary values (£18,000, 40,000 mi).
- Near misses on one field each: year 2017, Manual, Semi-Auto, 40,001 mi, £18,001. There's also an A3 Hybrid over budget for fuel filtering.
- 1 exact duplicate row (same as row 1), to test ID stability with duplicates.
- Q8 exists only in 2019 to 2020, so Q8 2015 to 2018 has zero rows (for `market_summary` on an empty range).
- 1 dirty row: A1 with mileage 3 and engineSize 0.0.

### LLM: the planned model is gone (open decision for Shrinidhi)

- `llama-3.3-70b-versatile` **no longer exists on Groq** (`model_not_found`, 404). Groq's current list: allam-2-7b, openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b, plus guard, TTS and whisper models.
- Smoke test (`scripts/smoke_llm.py --model <id>`, one dummy tool):
  - `openai/gpt-oss-120b`: **tool call returned.** 148 input and 61 output tokens (22 of them reasoning). Free-tier limits: 8,000 tokens/min, 1,000 requests/day.
  - `openai/gpt-oss-20b`: **tool call returned.** Same limits.
  - `qwen/qwen3.8-27b`: failed with 429. Its free tier caps output at 1,000 tokens/min and the client asks for 2,048 by default. It would need `max_tokens` ≤ 1000 and has much tighter pacing.
- Both working models passed `model="Audi A3"` rather than `"A3"`. The `search_listings` docstring and schema description need to say "model code only", and Lane A should normalise it (strip "Audi", whitespace, case).
- NIM fallback: **not tested**, because `NVIDIA_API_KEY` is empty in `.env`. The script supports `--nim` (model `meta/llama-3.3-70b-instruct`, base_url `https://integrate.api.nvidia.com/v1`).
- Suggested replacement: `openai/gpt-oss-120b` (the strongest listed model that does tool calling, with the same 8k TPM pacing assumption as the plan). Not adopted yet; Shrinidhi to confirm at CHECKPOINT 0.

### Other Phase 0 choices

- `data/raw/*.csv` is gitignored for now, per PLAN.md Phase 0 step 3. The licence allows committing it, so it's Shrinidhi's call.
- `.github/workflows/ci.yml` not created yet. Lane C owns it, and a workflow with zero tests would fail (pytest exits 5), so pushing Phase 0 would give a red CI run.
- Added `scripts/smoke_llm.py` (not in the PLAN.md layout) to keep the smoke test runnable.
- Nothing created under `eval/gold/` (Phase 1 is Shrinidhi's).

### Open questions for Shrinidhi (broker decisions, not Claude Code's)

1. Does "automatic" include `Semi-Auto`? It's 3,591 listings, more than `Automatic` itself. Options: map "automatic" to {Automatic, Semi-Auto}, or treat them as distinct and let the agent relax Automatic to Semi-Auto. The answer also shapes the relaxation policy and the gold `expected_constraints`.
2. Duplicates: keep all 103 duplicate rows as separate listings, or drop them in cleaning?
3. Which LLM replaces `llama-3.3-70b-versatile`?
4. Commit `audi.csv` (CC0 allows it), or keep it gitignored and document the download?
