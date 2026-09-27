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

### LLM: the planned model is gone (resolved at CHECKPOINT 0: gpt-oss-120b)

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

### Open questions for Shrinidhi (broker decisions, not Claude Code's): all resolved at CHECKPOINT 0, see below

1. Does "automatic" include `Semi-Auto`? It's 3,591 listings, more than `Automatic` itself. Options: map "automatic" to {Automatic, Semi-Auto}, or treat them as distinct and let the agent relax Automatic to Semi-Auto. The answer also shapes the relaxation policy and the gold `expected_constraints`.
2. Duplicates: keep all 103 duplicate rows as separate listings, or drop them in cleaning?
3. Which LLM replaces `llama-3.3-70b-versatile`?
4. Commit `audi.csv` (CC0 allows it), or keep it gitignored and document the download?

---

## CHECKPOINT 0 decisions (Shrinidhi, 2026-09-27)

### Transmission: "automatic" matches Automatic OR Semi-Auto
- **Decision:** the LLM passes `transmission: "Automatic"` for any "automatic" request. `search_listings` and the false-fit checker expand it in code to {Automatic, Semi-Auto}. `"Manual"` matches Manual only. `"Semi-Auto"` stays a valid schema value, but the prompt doesn't steer the model toward it.
- **Reason:** in UK listings, "Semi-Auto" is usually a dual-clutch gearbox with no clutch pedal, which is what a customer asking for an automatic wants. *This is inferred from domain knowledge, not verified from the data* (the dataset has no gearbox detail).
- **Why expand in code, not in the LLM:** it's one deterministic rule shared by search, checker and scoring, and it doesn't depend on the model remembering it.
- **Rejected:** treating Semi-Auto as distinct (it would hide 3,591 listings from "automatic" requests and make "relax to Semi-Auto" a routine relaxation); asking the LLM to pass a list of transmissions.

### Duplicates: drop exact duplicate rows at load time
- **Decision:** `data.py` drops exact duplicate rows (all 9 columns equal) before assigning listing IDs, and logs how many it dropped (103 expected on the raw file, 1 on `mini.csv`).
- **Reason:** identical rows are almost certainly the same advert scraped twice. Keeping them would fill the 5-result list with copies.
- **Rejected:** keeping duplicates as separate listings.

### LLM: Groq `openai/gpt-oss-120b`, reasoning_effort low
- **Decision:** the model is `openai/gpt-oss-120b` on Groq. It replaces `llama-3.3-70b-versatile`, which Groq has retired (404 `model_not_found`).
- **reasoning_effort:** supported. `langchain-groq` 1.1.3 has a `reasoning_effort: str | None` field on `ChatGroq`. Set it as `ChatGroq(model="openai/gpt-oss-120b", temperature=0, reasoning_effort="low")`. Checked with a live call: the API accepted `"low"`, returned a correct tool call, and echoed `reasoning_effort: low` in `response_metadata`. On the one-line smoke prompt, reasoning tokens were 21 with or without the setting, so this call showed no effect. Lane B puts this in `llm.py`.
- **Reason:** it's the strongest listed Groq model that returned a tool call in the smoke test, with the same 8,000 tokens/min free-tier limit PLAN.md paces for. Low effort keeps output tokens (which count toward the 8k/min) down.
- **Rejected:** `openai/gpt-oss-20b` (weaker); `qwen/qwen3.8-27b` (1,000 output tokens/min free-tier cap).
- **The NIM fallback (`meta/llama-3.3-70b-instruct` via `langchain-openai`) is untested:** `NVIDIA_API_KEY` is empty.
- Side note: with the docstring "model code (e.g. 'A3')", the model passed `model: "A3"`. With a vaguer docstring it passed `"Audi A3"`. Lane A should still normalise it.

### Commit `data/raw/audi.csv`
- **Decision:** commit it; removed from `.gitignore`.
- **Reason:** the licence is CC0: Public Domain (read on the Kaggle dataset page on 2026-09-27), and committing it makes the eval reproducible without a download.
- **Rejected:** gitignoring it and documenting the download.

### Numeric limits are inclusive
- **Decision:** every numeric limit (`year_min`, `year_max`, `price_max`, `mileage_max`) passes a listing whose value equals the limit. The gold set, `search_listings` and the checker all use this rule. "Under £18,000" is written `price_max: 18000`.
- **Reason:** one rule everywhere, with no off-by-one disagreements between the gold author, the tool and the checker.
- **Rejected:** strict "under".

### `ask_customer` is a run outcome
- **Decision:** added to the outcome values in PLAN.md section 1 and to outcome scoring in section 4. Vague gold tasks use `expected_outcome: "ask_customer"`. Already present in `schemas.RunOutcome`.

---

## Lane A: data, tools, checker (2026-09-27, continuous-mode run)

- **Listing IDs = row position in the raw file** (`L00000` = first data row), assigned *before* dropping duplicates, keeping the first copy. Why: stable and readable, and dropping a duplicate never shifts another listing's ID. Rejected: content hashes (unreadable, and identical rows collide); renumbering after de-duplication (IDs would change if the cleaning rule changed).
- **Loader fails loudly** on missing columns or on transmission/fuel values outside the schema, rather than silently dropping rows.
- **`search_listings` takes the constraint fields as flat arguments** (`args_schema = Constraints`), not a nested `constraints` object. Why: flat arguments are easier for the LLM to fill, and the tool-call args compare directly with gold `expected_constraints`. Rejected: nested `{"constraints": {...}}` as literally written in PLAN.md section 1.
- **Normalisation lives in the `Constraints` validators:** model `"Audi A3"` / `" a3"` → `"A3"`; transmission and fuel type match case-insensitively (`"automatic"` → `"Automatic"`). Consequence for scoring: constraint parsing is compared *after* normalisation, so `"Audi A3"` counts as `"A3"`. Why: the search behaves identically, so the metric should measure understanding, not string formatting. Rejected: strict raw-string comparison.
- **Ranking tie-break:** price asc, mileage asc, then listing_id asc, so ties are fully deterministic.
- **The checker is a separate row-wise implementation** of the same rules (inclusive limits, Automatic ⊇ Semi-Auto). A test cross-checks it against the pandas filter on every fixture row for 768 constraint combinations. Why: if search and checker shared one function, a bug in it would hide false fits. `accepted_transmissions()` is the one shared piece; it's a single-line rule.
- **An unknown listing ID in a `match` counts as a false fit** (reported as `unknown_listing`).
- **`submit_answer` validates consistency:** `relaxed_match` needs `relaxed_constraint`; others must not have one; `match`/`relaxed_match` need ≥1 ID; `no_match` needs none. Unknown IDs raise `ToolException`. Tools use `handle_tool_error=True`, so the model gets an error message and can retry within the step cap instead of the run crashing. The trace records the error. Rejected: accepting anything and relying only on scoring, which would let malformed answers end runs.
- **`market_summary` returns** count, price and mileage min/median/max, counts by transmission and fuel, and the year range that model exists in at all (`model_years_available`), which helps when the requested range is empty. Both tools add a `note` listing the known models when the model code is unknown.
- **Tool outputs are JSON strings**, so the ToolMessage content is exactly what's logged.
- **Tool descriptions** (what the LLM reads) are module constants in `tools.py`: `SEARCH_DESCRIPTION` etc.

---

## Lane B: graph, LLM factory (2026-09-27)

- **Graph shape:** nodes `agent`, `tools` (prebuilt `ToolNode`), `record`, `nudge`, `gave_up`. The routers are hand-written; `tools_condition` is too simple because it knows nothing about terminal tools or the cap. See CHECKPOINTS.md for the node-by-node explanation.
- **Step cap counted in state** (`llm_calls`, +1 per agent node), not via LangGraph's `recursion_limit`. Why: the cap is on LLM calls, and hitting it must record `gave_up`, not raise `GraphRecursionError`.
- **A terminal tool ends the run only if it succeeded.** A `submit_answer` rejected for bad arguments (validation or unknown IDs) returns an error ToolMessage and the agent gets another turn within the cap. Rejected: ending on any terminal *call*, which would record answers the tool refused.
- **Parallel tool calls are allowed.** If one AI message contains several calls, `record` processes them in order and stops at the first successful terminal one. Rejected: `parallel_tool_calls=False`, a provider-specific flag that the fake model can't exercise.
- **Text-only replies:** routed to `nudge`, which appends a clearly labelled "[Note from the system, not the customer]" HumanMessage and returns to `agent`. Every such turn counts toward the cap. Rejected: treating a text reply as immediate `gave_up` (too harsh for one slip); a mid-conversation SystemMessage (not all providers accept one).
- **`original_constraints` = the arguments of the first *successful* `search_listings` call.** The in-graph false-fit check runs only when outcome is `match` and a search happened. With no search, `false_fit` stays None. The eval scorer checks against the gold `expected_constraints` instead (see Lane C).
- **`build_graph(llm, *, listings=None, llm_caller=None)`:** `listings` defaults to the real data (tests pass `mini.csv`). `llm_caller(runnable, messages)` is a hook so the eval harness can add pacing, 429 retries and logging without the graph knowing about them.
- **`prompt_fingerprint`** is a sha256 of the system prompt plus every tool's name, description and argument schema, attached to the compiled graph. The harness records it in every results file, so "nothing changed after the test run" can be verified.
- **Relaxation policy is a marked placeholder** (`graph.RELAXATION_POLICY`, starting `[PLACEHOLDER - relaxation policy not yet supplied.]`). It carries an interim sentence ("loosen at most one constraint, and say which and by how much") so the smoke run can exercise the relaxed path. That sentence is not a broker policy.
- **Vagueness rule in the prompt:** ask when there's "no model and no concrete requirement such as budget, year, mileage or gearbox". This is my wording; the gold "too vague" tasks will show whether it matches Shrinidhi's judgement. Tune it on dev only.
- **LLM factory `make_llm(provider=None, *, max_retries=2)`:** Groq `openai/gpt-oss-120b`, `temperature=0`, `reasoning_effort="low"`, `max_tokens=1024`. The 1,024 cap bounds output (and reasoning) tokens per call, which count toward Groq's 8k tokens/min. `CARMATCH_LLM=nim` selects the untested NIM fallback. It loads `.env` from the repo root.
- **Fake model for tests:** `tests/fakes.py` `ScriptedChatModel(GenericFakeChatModel)` with a no-op `bind_tools`, plus `call()`/`say()` helpers that attach `usage_metadata`.

---

## Lane C: scoring, harness, CI (2026-09-27)

- **Scoring works on normalised constraints**, and a missing field equals an explicit null. An *extra* constraint the customer didn't state counts as wrong. Invalid arguments count as wrong. Only the *first* `search_listings` call is scored (PLAN.md section 4), whether or not it succeeded.
- **Metrics that don't apply are None and excluded from the denominator.** Examples: constraint parsing on tasks with no expected search; relaxed-constraint correctness on tasks not expected to be `relaxed_match`. Every metric is reported as "k of n" with its own n. `gave_up` and harness errors (outcome None) never equal a gold outcome.
- **False fit is judged against the gold `expected_constraints`**, not the agent's own parse. Why: if the agent drops a constraint when searching and then calls the results a match, that's exactly the failure to catch. Fallback when a task has no gold constraints: the agent's first search. The in-graph check (against the agent's parse) is kept in the results as `in_graph_false_fit`. Semi-Auto satisfies "Automatic" here too (same rule).
- **Extra logged count, not a headline metric:** `relaxed_extra_violation`, a `relaxed_match` whose listings break a gold constraint *other than* the one named as relaxed.
- **Pacing:** a sliding 60-second token window with a budget of 7,200 tokens/min (10% under Groq's 8,000). The estimate before each call is prompt characters ÷ 4 + 1,200 (system prompt and tool schemas) + 400 (expected output). After the call, actual usage replaces the estimate in the window. An underestimate is caught by the 429 retry. Rejected: LangChain's `InMemoryRateLimiter`, which limits requests, not tokens.
- **429 retries in the harness, not the client:** the eval builds the LLM with `max_retries=0`. `EvalCaller` retries 429s up to 6 times with backoff: the `retry-after` header if present, else 2, 4, 8… up to 60 s. Every retry is logged. Other exceptions aren't retried. Why: the retries need to be visible in the logs and the tests.
- **Structured logs:** a plain JSON-lines file per run (`EventLog`), with events `run_start`, `task_start`, `llm_call` (tokens, reasoning tokens, estimate, latency, wait, retries, tool names), `llm_retry_429`, `llm_error`, `tool_call` (name, args, status, result), `task_end`, `run_end`. Tool calls are written from the trace after each task. Rejected: Python `logging` with a JSON formatter (more setup for the same output, and no new dependencies allowed).
- **A crash inside a task is recorded, not raised:** the task gets `error` and scores as wrong, and the run continues.
- **Latency:** `latency_s` is task wall-clock *including* pacing waits. `llm_s` (time inside LLM calls) and `waited_s` are recorded separately, because pacing inflates wall-clock on the free tier.
- **Guards for the gold set (code-level):**
  - `--split dev|test` reads `eval/gold/<split>.jsonl` only if `eval/gold/FROZEN.md` exists and a line naming that file contains its current sha256.
  - `--split test` also needs `--confirm-test-split` and refuses if any `eval/results/test_*.json` already exists.
  - `--tasks` refuses paths under `eval/gold/`, so the hash check can't be bypassed.
- **Every results file records** the model, reasoning effort, git commit and a dirty flag, the tasks file and its sha256, and the prompt fingerprint.
- **Run as `python -m eval.run_eval`** from the repo root. Added `eval/__init__.py` and `pythonpath = ["."]` in the pytest config so tests import `eval.*`. `eval` isn't an installed package, which is deliberate because it's not part of the library.
- **Added `tests/test_run_eval.py`** (not in the PLAN.md layout): pacing, retries, guards, and an end-to-end run with the scripted fake model on `mini.csv`.
- **CI:** GitHub Actions on push and pull request, ubuntu-latest, Python 3.11, `pip install -e ".[dev]"`, `pytest -m "not slow"`. No secrets needed.

---

## Harness fixes from the smoke run (2026-09-27)

These fix harness infrastructure. The system prompt and tool descriptions were **not** changed; the prompt fingerprint is still `eca9cd476398…`.

- **Keep the partial trace when a run raises.** The harness now uses `graph.stream(..., stream_mode="values")` and keeps the last state. Motivation: in smoke run 1, s02 raised on its 4th LLM call and the whole trace was lost, so constraint parsing was wrongly scored "no" even though the search had the right arguments. Rejected: `invoke()` plus re-reading messages from logs.
- **Retry Groq `400 tool_use_failed` up to 2 times.** Each retry is logged (`llm_retry_tool_use_failed`), counted per task and in the summary table, and its estimated tokens are added to the pacing window. Motivation: in smoke run 1, gpt-oss-120b sent a tool call named `json`, which Groq rejects before our code sees it. Why retry: it's a model output-format glitch, the retry is visible in every report, and it can't turn a wrong answer into a right one. Rejected: no retry (one glitch kills the task), and silent retry (would hide the failure). **The same error recurred on both retries in smoke run 2** (tool named `commentary`), so at temperature 0 the retry doesn't rescue this case. See the open question in CHECKPOINTS.md.
- **The pacer now counts tokens from failed generations.** Motivation: smoke run 2 hit one 429 on s03 right after s02's three failed attempts, which the pacer hadn't counted.

---

## Diagnosis run: harness additions (2026-09-27)

- **The raw failed output is logged untruncated.** When the API rejects a call, `llm_retry_tool_use_failed` and `llm_error` events now carry the provider's error body: `status_code`, `code`, `message`, and `failed_generation` (Groq's copy of exactly what the model tried to emit). It's read from the exception's `body` dict, with no string parsing. Before this, the error was a repr cut at 500 characters.
- **Diagnosis-only overrides:** `run_eval` gains `--ids`, `--model`, `--reasoning-effort` (`unset` sends none), `--max-tokens` and `--label`. `make_llm` gains matching keyword arguments whose defaults are the chosen configuration, so default behaviour is unchanged (a test covers this). The overrides are **refused with `--split`**, so gold runs always use the committed configuration. Every results file records the model, effort, max_tokens and task IDs actually used. Rejected: editing `llm.py` constants for each setup, which risks leaving a non-default model committed.
- Ran step 5 (logging) before step 4 (the comparison runs) so the step 4 logs contain the raw output.
