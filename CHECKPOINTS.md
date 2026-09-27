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

---

## Step 3: Lane C, scoring, harness and CI

**Built**
- `eval/score.py`: pure `score_task`, `summarize`, `summary_markdown`, plus helpers.
- `eval/run_eval.py`:
  - `TokenPacer` keeps usage under a token budget per rolling minute.
  - `EvalCaller` is the graph's `llm_caller`: it paces, retries 429s, logs, and counts tokens and latency per task.
  - `EventLog` writes the JSON-lines log.
  - Gold guards: `check_frozen` and `check_test_run_allowed`.
  - `run_tasks`, `write_results`, and a CLI.
- `.github/workflows/ci.yml`.
- Tests: `test_score.py` (12 tests), `test_run_eval.py` (10 tests). **73 tests pass in total.**

**Design choices:** see DECISIONS.md, section "Lane C". The important ones:
- False fit is judged against the gold constraints.
- Each metric's denominator counts only the tasks it applies to.
- Pacing is by tokens, not requests, and 429 retries are logged.
- The frozen-hash check and the run-once rule for the test split are enforced in code.

**Failures:** none in this step. I didn't open or create anything under `eval/gold/`; the guard tests use a temporary directory.

---

## Step 4: integration test

**Built:** `tests/test_integration.py`, marked `slow` and skipped if `GROQ_API_KEY` is unset. It runs one real agent run on the real Audi data with a request written in the test file ("2019 or newer Audi Q3, automatic, under 30,000 miles, budget £25,000"; 3 real listings fit). It checks the pipeline, not answer quality: the LLM call count is within the cap, the outcome is valid, a search happened, and the parsed model is Q3.

**What happened:** the slow test ran **three times**, not once.
- Run 1: the agent worked (search, then submit `match` with 3 IDs), but the test crashed in its own `print`. The Windows console (cp1252) can't encode the non-breaking hyphen (U+2011) the model wrote in its message.
- Run 2: my first fix script failed its own match check, and because I'd chained the commands without `&&`, pytest reran with the old code and hit the same crash. My mistake.
- Run 3 (after fixing the print with `ascii()`): **passed.** 2 LLM calls. The first search's args were `{model: Q3, year_min: 2019, transmission: Automatic, mileage_max: 30000, price_max: 25000}`; then `submit_answer` with `match` and `L08494, L08499, L07328`.

**Also fixed:** `run_eval.py` now reconfigures stdout with `errors="replace"`, so the same console problem can't crash an eval run while it prints its summary. The results files are written as UTF-8 regardless.

---

## Step 5: smoke run (pipeline check only, not an evaluation)

`eval/smoke/smoke.jsonl` holds 3 requests invented by Claude Code, not gold and not written by the broker:
- s01: an A1 clean match (4 real listings fit).
- s02: an A3 manual, 2019 or newer, under 10k miles and up to £15,000. Nothing fits; relaxing price alone finds cars from £15,700.
- s03: a vague "reliable family car".

**Smoke run 1** (`smoke_20260927-094850`, commit `baaee16`, marked dirty because `smoke.jsonl` wasn't committed yet):
- s01 and s03 ended as expected.
- s02 errored: on the 4th LLM call, gpt-oss-120b emitted a tool call named `json` and Groq rejected it (400 `tool_use_failed`).
- The trace was lost, a harness bug that is now fixed.

**Smoke run 2** (`smoke_20260927-095107`, commit `a9f28a8`, clean), after the harness fixes:
- s01 `match` (2 calls). s03 `ask_customer` (1 call).
- s02: search (0 found) → market_summary → search with `price_max` 16000 (3 found) → the submit attempt came out as a tool named `commentary`, and was rejected 3 times (original plus 2 retries) → recorded as an error. The partial trace was kept and constraint parsing scored correctly.
- Token use: 6 successful LLM calls, 7,974 input and 499 output tokens (failed attempts not included in these counts). One 429 retry and 2 malformed-tool-call retries. 66 s wall-clock, of which about 59 s was a pacing wait inside s02.

**What this shows:**
- The pipeline works end to end: real tool calling, all four tools, terminal routing, scoring, logs, pacing and 429 retry.
- One real problem: **gpt-oss-120b on Groq can emit its internal channel names (`json`, `commentary`) as tool names**, here when submitting after a relaxation. It reproduced on every attempt at temperature 0.

**For Shrinidhi to decide** (not changed here, to avoid tuning on non-dev data): whether to handle this in Phase 3 using dev failures. Options:
- (a) Leave it and count it as a failure.
- (b) Try `reasoning_effort="medium"`.
- (c) Try the NIM fallback or another model.
- (d) Accept a small prompt or description change motivated by a dev failure.

**After this run:** the pacer now also counts the tokens used by failed generations. I didn't rerun the smoke set after that change.

---

## Steps 6 and 7: tests and README

- `pytest -m "not slow"`: **75 passed**, 1 deselected (the slow test).
- The slow integration test ran 3 times in total (see step 4). The final run passed.
- `README.md` drafted, with sections for what it does, how to run it, the manual-first method, results (left as "TODO: pending frozen gold set evaluation") and limitations. It makes no accuracy or success claims. The smoke run is described as a pipeline check, and the malformed-tool-call behaviour is listed as a limitation.

---

## Step 8: push and CI

- Pushed `main`, `6944d92..c690249`, no force.
- GitHub Actions run 36312384066 (`ci`, job `unit-tests`) **passed** in 28 s: install plus `pytest -m "not slow"` on ubuntu-latest, Python 3.11.
- Two warnings from GitHub, neither a failure: `actions/checkout@v4` and `actions/setup-python@v5` target the deprecated Node 20; `ubuntu-latest` moves to Ubuntu 26 from 2026-10-19.

---

# Malformed-tool-call diagnosis run (2026-09-27)

## Step 1: re-run of the existing smoke set (`smoke_20260927-100056`, commit `9c314d8`, clean)

Same three tasks as before. The harness fixes from the last run work as intended:
- **Trace kept on failure:** s02 errored on its 4th call, and its partial trace (search_listings, market_summary, search_listings) is in the results. Constraint parsing scored 2 of 2.
- **Retries counted:** s02 shows `tool_use_failed_retries: 2`. The summary table shows 2.
- **Failed-attempt tokens in the rate window:** s03 waited 57 s before its call, and there were 0 HTTP 429 retries. The previous run, before this fix, had one 429 at the same point.

The s02 outcome is unchanged: the malformed terminal call recurred on all 3 attempts. s01 `match`, s03 `ask_customer`. 6 successful LLM calls, 7,974 input and 498 output tokens, 121 s wall-clock (most of it pacing waits).

## Step 2: five new smoke tasks

Appended to `eval/smoke/smoke.jsonl`. **New: s04, s05, s06, s07, s08.** The originals s01 to s03 are unchanged. All were invented by Claude Code and checked against the CSV; each `broker_notes` says so. They are not gold.

| id | request (short) | exact matches | what a single relaxation gives | expected (guess) |
|---|---|---|---|---|
| s04 | Q5, 2019+, automatic, <20k mi, £28,000 | 0 | price alone: from £28,070 (0.25% over); year alone: newest 2018 | relaxed_match, price_max |
| s05 | TT, 2018+, automatic, ≤£20,000 | 0 | year_min 2017: 35 cars; price alone: from £22,300 (11.5% over) | relaxed_match, year_min |
| s06 | A1, 2020, automatic, ≤£14,000 | 0 | newest within budget is 2018; price alone: from £20,495 (46% over) | relaxed_match, year_min |
| s07 | R8, 2018+, <£15,000 | 0 | cheapest R8 of any year £33,950 | no_match |
| s08 | Q8, <£25,000 | 0 | cheapest Q8 £48,022 | no_match |

The expected relaxed constraints are guesses, because the real relaxation policy hasn't been written yet. The diagnosis in step 4 counts clean versus malformed terminal calls, so it doesn't depend on those guesses.

Relaxation and no-match tasks used in step 4: **s02, s04, s05, s06, s07, s08** (6 tasks).

## Step 3: Groq models available via the API

From `GET /models` (the `supported_features` field) and response rate-limit headers:

| model | supported_features | tokens/min | requests/day |
|---|---|---|---|
| openai/gpt-oss-120b | tools, json_mode, structured_outputs, reasoning | 8,000 | 1,000 |
| openai/gpt-oss-20b | tools, json_mode, structured_outputs, reasoning | 8,000 | 1,000 |
| openai/gpt-oss-safeguard-20b | tools, json_mode, structured_outputs, reasoning | 8,000 | 1,000 |
| qwen/qwen3.8-27b | tools, json_mode, reasoning | 8,000 | 1,000 (plus a 1,000 output-tokens/min cap, seen in a 429 in Phase 0; not in the headers) |
| allam-2-7b | json_mode | 6,000 | 7,000 |
| meta-llama/llama-prompt-guard-2-86m | json_mode | - | - |
| meta-llama/llama-prompt-guard-2-22m, whisper-large-v3(-turbo), canopylabs/orpheus-* | none listed (guard, speech-to-text, text-to-speech) | - | - |

Four models claim tool calling: gpt-oss-120b, gpt-oss-20b, gpt-oss-safeguard-20b (a safety-classifier variant) and qwen3.8-27b.

## Step 5 (done before step 4): raw failed output in the log

`llm_retry_tool_use_failed` and `llm_error` log events now include Groq's untruncated `failed_generation`, plus `code` and `message`. There are also diagnosis-only overrides (`--model`, `--reasoning-effort`, `--max-tokens`, `--ids`, `--label`), which are refused for gold splits. The default model is unchanged. 77 unit tests pass.

## Step 4 and 6: terminal-call diagnosis on 6 relaxation/no-match smoke tasks

Tasks s02, s04, s05, s06, s07 and s08, one run each per setup. The system prompt and tool descriptions are unchanged; the prompt fingerprint is `eca9cd476398` in all four runs. The counts come from `python -m eval.diagnose_terminal_calls <results.json ...>`, which reads each run's results JSON and JSON-lines log.

| setup | model | reasoning_effort | tasks run | clean terminal calls | malformed calls (provider-rejected) | gave_up | errors | total tokens (in + out) | time (s) |
|---|---|---|---|---|---|---|---|---|---|
| A | openai/gpt-oss-120b | low | 6 | 2 | 6 (5 `tool_use_failed`, 1 `output_parse_failed`), in 3 tasks | 2 | 2 | 40,492 (38,177 + 2,315) | 428.5 |
| B | openai/gpt-oss-120b | medium | 6 | 5 | 0 | 1 | 0 | 45,010 (39,464 + 5,546) | 429.4 |
| C | openai/gpt-oss-20b | low | 6 | 6 | 0 | 0 | 0 | 35,797 (33,701 + 2,096) | 304.8 |
| D | qwen/qwen3.8-27b | not set (model default); max_tokens 1000 | 6 | 5 | 0 | 1 | 0 | 57,798 (55,711 + 2,087) | 547.3 |

How the columns are counted:
- **clean terminal calls:** tasks that ended with a successful `submit_answer` or `ask_customer`.
- **malformed calls:** every attempt Groq rejected with 400 `tool_use_failed` or `output_parse_failed`. That includes the harness's retries: up to 2 per call, for `tool_use_failed` only.
- **tokens:** successful LLM calls only.
- **time:** summed task wall-clock, including pacing waits.
- Terminal calls rejected by our own schema: 0 in every setup.

**Conditions:**
- D used `max_tokens=1000` because of qwen's free-tier cap of 1,000 output tokens per minute. It hit 1 HTTP 429 retry and fit within the limits.
- A ran at the same time as C and D; B ran alone after A.
- HTTP 429 retries: A 5, B 0, C 0, D 1.

**Outcome per task** (facts, not scored against the guessed expectations):

| task | A (120b low) | B (120b medium) | C (20b low) | D (qwen) |
|---|---|---|---|---|
| s02 | error (malformed ×3) | relaxed_match, price_max | relaxed_match, price_max | no_match |
| s04 | match | relaxed_match, price_max | relaxed_match, transmission | relaxed_match, price_max |
| s05 | relaxed_match (after 1 malformed retry) | relaxed_match, year_min | relaxed_match, transmission | gave_up |
| s06 | gave_up | gave_up | no_match | no_match |
| s07 | error (1 malformed, then `output_parse_failed`) | relaxed_match, price_max | no_match | no_match |
| s08 | gave_up | relaxed_match, price_max | no_match | no_match |

**gave_up traces, in brief:**
- A s06: 5 searches with varied limits, no terminal call.
- A s08: 4 tool calls and 2 replies with no tool call.
- B s06: 5 searches with varied price limits.
- D s05: after market_summary, 4 identical searches with `model: null, transmission: null`.

**Examples of raw malformed output** (Groq's `failed_generation`, from setup A's log `smoke_A-120b-low_20260927-182200.log.jsonl`):

1. s02, `tool_use_failed` ("attempted to call tool 'json'"). The arguments are a complete submit_answer payload under the tool name `json`:
   ```
   {"name": "json", "arguments": {
     "listing_ids": ["L00045", "L07562", "L00074"],
     "message": "I couldn't find any A3s that meet all your criteria at £15,000, but by extending the budget slightly to £16,000 you get three options that match everything else.",
     "outcome": "relaxed_match",
     "relaxed_constraint": "price_max"
   }}
   ```
2. s07, `tool_use_failed` ("attempted to call tool 'commentary'"). Again a submit_answer payload, here offering an R8 at £93,950 for a £15,000 budget:
   ```
   {"name": "commentary", "arguments": {
     "listing_ids": ["L04391"],
     "message": "I couldn't find an R8 from 2018 or newer under £15,000, but there is one from 2018 priced at £93,950, which meets all your other criteria.",
     "outcome": "relaxed_match",
     "relaxed_constraint": "price_max"
   }}
   ```
3. s07 retry, `output_parse_failed` ("The model generated output that could not be parsed"). The whole generation was:
   ```
   We relaxed price to min. Provide relaxed_match.
   ```

In all 6 provider-rejected attempts in A, the model was trying to submit the final answer; none were search or market_summary calls. In examples 1 and 2, the arguments would have passed our `submit_answer` schema if the tool name had been correct.
