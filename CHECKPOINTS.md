# Checkpoint reports

Appended in order. Plain English: what was built, each design choice and why, and anything that failed.

---

## Step 0: worktrees removed (2026-09-27)

Removed the three unused worktrees (`../cm-tools`, `../cm-graph`, `../cm-eval`) and deleted the `lane/tools`, `lane/graph` and `lane/eval` branches. All three were clean and at `6944d92`, the same commit as `main`, so no work was lost. From here, Lanes A, B and C run in sequence directly on `main`, in one session.

---

## Step 1: Lane A, data, tools and checker

**Built**
- `data.py`: `load_listings(path)` reads the CSV, strips whitespace, renames `fuelType`/`engineSize`, rejects unexpected values, assigns IDs from raw row position, and drops exact duplicates (logs the count). Real file: 10,668 rows, 103 duplicates dropped, 10,565 listings.
- `checks.py`: `accepted_transmissions`, `violations(listing, constraints)`, `fit_report`, `is_false_fit`.
- `tools.py`: `build_tools(listings)` returns the four LangChain tools. There's also a pure `search()` function used by the tool and the tests.
- `schemas.py`: added normalising validators, tool-argument descriptions, `SubmitAnswerArgs` consistency rules, and `false_fit` fields on `AgentState`.
- Tests: `test_data.py`, `test_tools.py`, `test_checks.py` plus `conftest.py`. **39 passed.**

**Design choices:** see DECISIONS.md, section "Lane A". In short: raw-row IDs, flat search arguments, normalisation in the schema, a deterministic tie-break, a checker kept deliberately separate from search and cross-checked, and tool errors returned to the model rather than crashing.

**What failed along the way:** one test had a hand-counted expectation wrong (A3 2018–2019 in `mini.csv` is 8 rows, not 7). The test was fixed; the code was right.

**Data note:** with the "automatic includes Semi-Auto" rule, the PLAN.md example request matches 9 real listings (it was 3 with Automatic only).

---

## Step 2: Lane B, graph and LLM factory

**Built:** `llm.py` (`make_llm`), `graph.py` (`build_graph`, `initial_state`, `SYSTEM_PROMPT`, `RELAXATION_POLICY` placeholder, `prompt_fingerprint`), `tests/fakes.py`, and `tests/test_graph_routing.py` (12 tests). Real Lane A tools are used, not stubs. **51 tests pass in total.**

**The graph in plain English**
- **START → agent.** The run begins with the customer's request as the only message.
- **agent:** makes one LLM call with the system prompt plus the conversation so far, with the four tools bound, and adds 1 to `llm_calls`.
- **agent → tools** if the reply contains tool calls.
- **agent → nudge** if the reply has no tool call and fewer than 6 calls have been made. `nudge` adds a labelled system note ("finish by calling submit_answer or ask_customer") and goes back to **agent**.
- **agent → gave_up** if the reply has no tool call and 6 calls have been made.
- **tools:** runs every requested tool. Bad arguments and unknown listing IDs come back as error messages to the model; they don't crash the run.
- **tools → record.** `record` looks at the round that just ran. It saves the first successful search's constraints as `original_constraints`. If `ask_customer` or `submit_answer` succeeded, it stores the outcome, listing IDs, relaxed constraint and message. For `match`, it runs the false-fit check against `original_constraints`.
- **record → END** if an outcome was recorded. **record → gave_up** if not, and 6 calls have been made. Otherwise **record → agent** for another turn.
- **gave_up:** sets outcome `gave_up`, then **END**.

**Tests cover:** search then submit; empty search → market_summary → relaxed search → relaxed_match (with `original_constraints` = the first, tight search); ask_customer ending after 1 call; endless searching stopped at exactly 6 calls with `gave_up`; text-only replies nudged then stopped; nudge then recover; a terminal call on the 6th call still counting; a match containing a non-fitting listing flagged as false fit; an invalid submit not ending the run; an unknown ID rejected then retried; the `llm_caller` hook wrapping every call; the placeholder and honesty rule present in the prompt.

**Failures:** none in this step.
