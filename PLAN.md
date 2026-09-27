# car-match-agent: build plan

Owner: Shrinidhi KJ. Purpose: a small, tested, honestly evaluated LangGraph agent that matches a customer's used-car request to listings, built "manual first": the expected answers are written by hand and frozen before the agent runs on them.

This file is the single source of truth. Every Claude Code session reads it (and CLAUDE.md) before doing anything.

---

## 0. How to run this plan

- Launch every session with `claude --permission-mode auto` (auto mode: no routine prompts, background safety checks). Do not use `--dangerously-skip-permissions` on this machine.
- Phases run in order. Inside Phase 2, the three lanes run in parallel, each in its own git worktree and its own terminal.
- Each phase ends with a CHECKPOINT: the session stops, writes a short report, and waits. Read the report and the diff before starting the next phase.
- Usage: three parallel sessions use the Pro 5-hour window about three times faster. If you're close to the limit, run the lanes one at a time instead.

---

## 1. What we are building (fixed decisions)

**Input:** a plain-English request, e.g. "a 2018 or newer Audi A3, automatic, under 40k miles, under £18,000".

**Data:** Kaggle "100,000 UK Used Car Data set", Audi file only (`audi.csv`). Scraped around 2020, so no trims, colours, locations or dealers, and prices are ~2020 listings. Stored at `data/raw/audi.csv` and committed: the licence is CC0: Public Domain (see DECISIONS.md). Cleaning at load time drops exact duplicate rows (103 in the raw file) and logs how many were dropped.

**Tools (the model chooses between these):**
- `search_listings(constraints)`: deterministic filter and rank in pandas. Ranking rule: price ascending, then mileage ascending. Returns at most 5 listings with stable IDs. The LLM never ranks cars itself.

**Matching rules (shared by `search_listings`, the false-fit checker and the gold set):**
- All numeric limits are inclusive: a listing whose value equals the limit passes (`mileage_max: 40000` keeps a 40,000-mile car).
- Transmission: a customer's "automatic" is passed by the LLM as `"Automatic"` and expanded in code to match Automatic OR Semi-Auto. `"Manual"` matches Manual only.
- `market_summary(model, year_min, year_max)`: count of listings plus price and mileage spread for that model and year range. Used when a search comes back empty, to work out which constraint is blocking.
- `ask_customer(question)`: terminal. Ends the run with a clarifying question.
- `submit_answer(outcome, listing_ids, relaxed_constraint, message)`: terminal. `outcome` is one of `match`, `relaxed_match`, `no_match`.

**Stopping conditions:** the run ends when a terminal tool is called, or when the step cap (6 LLM calls) is hit, in which case outcome is recorded as `gave_up`.

**Run outcomes:** `match`, `relaxed_match`, `no_match` (from `submit_answer`), `ask_customer` (the run ended with `ask_customer`), and `gave_up` (step cap hit). The agent can only submit the first three.

**Honesty rule built into the code:** if the agent submits `match`, a deterministic check verifies every returned listing satisfies every original constraint, using the same matching rules as `search_listings`. A violation is counted as a **false fit**, the key failure this project exists to catch.

**LLM:** Groq, `openai/gpt-oss-120b` via `langchain-groq`, with `reasoning_effort="low"` (`llama-3.3-70b-versatile` was retired from Groq; see DECISIONS.md). Fallback: NVIDIA NIM via `langchain-openai` with a custom `base_url`, untested (no key yet). Groq free tier is ~8,000 tokens per minute including output, so the eval harness paces requests and retries on 429.

**Out of scope today:** other manufacturers, live stock, funders, brokers, UI, cloud deploy. Stretch only after Phase 4: Langfuse tracing.

---

## 2. Repo layout (created in Phase 0, then fixed)

```
car-match-agent/
  CLAUDE.md                 standing rules for every session
  PLAN.md                   this file
  DECISIONS.md              short log: decision, reason, alternative rejected
  pyproject.toml            deps + pytest config (slow marker)
  .env.example              GROQ_API_KEY=, NVIDIA_API_KEY=  (real .env gitignored)
  data/raw/audi.csv         Kaggle download, committed (CC0)
  src/carmatch/
    schemas.py              Pydantic: Constraints, Listing, Outcome, AgentState
    data.py                 load + clean CSV, assign stable listing IDs
    tools.py                the four tools
    graph.py                LangGraph build_graph(llm) -> compiled graph
    llm.py                  model factory (Groq / NIM) from env
    checks.py               false-fit checker
  eval/
    gold/dev.jsonl          5 tasks, may be used for prompt tuning
    gold/test.jsonl         15 tasks, run ONCE at the end, never tuned on
    gold/FROZEN.md          sha256 of both files + commit hash + date
    run_eval.py             runs agent over a split, writes results
    score.py                scoring, pure functions
    results/                JSON per run + summary markdown
  tests/
    fixtures/mini.csv       ~30 hand-written rows, used by all unit tests and CI
    test_data.py
    test_tools.py
    test_checks.py
    test_graph_routing.py   fake LLM, deterministic
    test_score.py
    test_integration.py     real LLM, @pytest.mark.slow
  .github/workflows/ci.yml  pytest -m "not slow" on every push
```

Tests and CI use only `tests/fixtures/mini.csv`, never the Kaggle file, so CI never needs a download or an API key.

---

## 3. Gold set format (Shrinidhi writes this by hand in Phase 1)

One JSON object per line:

```json
{"id": "t01",
 "request": "2018 or newer A3, automatic, under 40k miles, max £18,000",
 "expected_constraints": {"model": "A3", "year_min": 2018, "transmission": "Automatic", "mileage_max": 40000, "price_max": 18000},
 "expected_outcome": "match",
 "expected_relaxed": null,
 "expected_tools": ["search_listings", "submit_answer"],
 "broker_notes": "why I'd handle it this way"}
```

`expected_constraints` follows the matching rules in section 1: limits are inclusive ("under 40k miles" is `"mileage_max": 40000`), and "automatic" is written `"transmission": "Automatic"`. Tasks too vague to search have `"expected_outcome": "ask_customer"` and `"expected_tools": ["ask_customer"]`.

Target mix across 20 tasks: roughly 8 clean matches, 6 needing one constraint relaxed, 3 where nothing is close, 3 too vague to search. Split: 5 into dev, 15 into test, with every category present in test.

Decision Shrinidhi must make and write in `broker_notes` and DECISIONS.md: **the relaxation policy** (e.g. "relax mileage before year, never go more than 10% over budget"). The agent's system prompt states the same policy. This is your judgement as the broker, not Claude Code's.

---

## 4. Scoring (all deterministic, in `eval/score.py`)

Per task:
- **Outcome correct:** final outcome equals `expected_outcome`, one of `match`, `relaxed_match`, `no_match`, `ask_customer` (`gave_up` is always wrong).
- **Tool selection correct:** set of tools called equals `expected_tools` (order ignored, repeats ignored).
- **Constraint parsing correct:** arguments of the first `search_listings` call equal `expected_constraints` (only for tasks where a search is expected).
- **Relaxed constraint correct:** for `relaxed_match`, the named constraint equals `expected_relaxed`.
- **False fit:** outcome `match` but a returned listing breaks an original constraint.
- Also logged: LLM calls, input and output tokens, wall-clock latency.

Report as counts ("11 of 15"), never percentages or "high accuracy". No significance claims on 15 items.

---

## 5. Phases

### Phase 0: setup (one session, you watching loosely)

Before this, you do by hand: create an empty public GitHub repo `car-match-agent`, clone it, download `audi.csv` into `data/raw/`, put your keys in `.env`, and read the dataset's licence on its Kaggle page.

Prompt for Claude Code:

> Read PLAN.md fully. Before writing anything, check the current state of this repo (git status, existing files) and report it. Then do Phase 0 only:
> 1. Create the layout in PLAN.md section 2 (empty modules with docstrings are fine), pyproject.toml with deps: langgraph, langchain-core, langchain-groq, langchain-openai, pandas, pydantic, python-dotenv, pytest. Install into a venv. Record the exact installed versions in DECISIONS.md.
> 2. Check the installed LangGraph version's actual API (read the installed package or its docs) before writing any graph code later; note the correct imports for StateGraph, ToolNode, conditional edges and a fake chat model for tests in DECISIONS.md.
> 3. Inspect data/raw/audi.csv: columns, dtypes, row count, value ranges for year, price, mileage, the distinct model and transmission values, any dirty values (e.g. leading spaces in model names). Write findings to DECISIONS.md. Do not commit the CSV.
> 4. Write src/carmatch/schemas.py with the Pydantic models in PLAN.md sections 1 and 3, and write tests/fixtures/mini.csv (~30 rows, invented but realistic, same columns) that covers: exact matches, near misses on each constraint, and a model with zero rows in some year range.
> 5. Smoke test: a script that binds one dummy tool to the Groq model via langchain-groq and prints the tool call it returns. Confirm the model name is currently available. If tool calling fails, try the NIM fallback and report.
> 6. Write CLAUDE.md with the standing rules from PLAN.md section 6.
> Commit with clear messages. Do not push. Stop and write a CHECKPOINT report: what exists, what worked, what failed, anything in PLAN.md that the data contradicts.

**CHECKPOINT 0:** you check the smoke test actually made a tool call, the schemas look right, and the data matches the plan. Push.

### Phase 1: gold set (YOU, by hand, no Claude Code)

Runs while the Phase 2 lanes run. Browse the real CSV (Excel is fine) and write the 20 tasks in section 3 format, splitting 5/15. Write your relaxation policy. Then freeze: commit, compute `sha256` of both files, record them plus the commit hash and time in `eval/gold/FROZEN.md`, commit again. From here the gold files never change; if one is wrong, note it in the README rather than edit it.

### Phase 2: three parallel lanes

Set up worktrees from the repo root:

```
git worktree add ../cm-tools  -b lane/tools
git worktree add ../cm-graph  -b lane/graph
git worktree add ../cm-eval   -b lane/eval
```

Open a terminal in each and run `claude --permission-mode auto`. Each lane only touches its own files, so merges are clean.

**Lane A, data and tools.** Prompt:

> Read PLAN.md and CLAUDE.md. Check the current state of this worktree first. You own only: src/carmatch/data.py, tools.py, checks.py, tests/test_data.py, test_tools.py, test_checks.py. Implement data loading and cleaning (based on the findings in DECISIONS.md), stable listing IDs, the four tools as LangChain tools with clear docstrings (the docstrings are what the model reads when choosing a tool, so make them precise about when to use each one), and the false-fit checker. Unit test everything against tests/fixtures/mini.csv, including: empty results, each constraint filtering correctly, ranking order, market_summary on an empty range, and the checker catching a listing that breaks one constraint. No LLM calls in this lane. Run pytest until green. Commit. Stop with a CHECKPOINT report listing each design choice and why.

**Lane B, graph and routing.** Prompt:

> Read PLAN.md and CLAUDE.md, and the LangGraph API notes in DECISIONS.md. Check the current state of this worktree first. You own only: src/carmatch/graph.py, llm.py, tests/test_graph_routing.py. Build build_graph(llm) with: an agent node (LLM with the four tools bound), a tool node, a conditional edge that routes to tools when there are tool calls, ends after a terminal tool (ask_customer or submit_answer), and routes to a forced stop recording outcome gave_up when the 6-call cap is hit. Tool functions may be stubs importing the signatures from schemas.py, since Lane A is writing the real ones. The system prompt must state the relaxation policy placeholder from PLAN.md section 3 and forbid presenting a non-fitting car as a match. Tests use a fake chat model with scripted responses, no network: search then submit ends correctly; empty search then market_summary then relaxed submit; ask_customer ends immediately; a model that never calls a terminal tool is stopped at the cap with gave_up; the step count is correct. Run pytest until green. Commit. Stop with a CHECKPOINT report that explains the graph in plain English, node by node and edge by edge.

**Lane C, eval harness and CI.** Prompt:

> Read PLAN.md and CLAUDE.md. Check the current state of this worktree first. You own only: eval/run_eval.py, eval/score.py, tests/test_score.py, .github/workflows/ci.yml, and the slow-marker config in pyproject.toml. Do NOT open eval/gold/test.jsonl, in any way, including through shell commands. Write score.py as pure functions implementing PLAN.md section 4, tested on small invented examples in the test file. Write run_eval.py: takes --split dev|test, runs a compiled graph over each task, records the full tool-call trace, tokens and latency per task, paces calls to stay under 8,000 tokens per minute and retries on HTTP 429 with backoff, writes eval/results/<split>_<timestamp>.json and a summary markdown table of counts. Structured logging (JSON lines) for each LLM call and tool call. Test it end to end with a fake graph. Write ci.yml: on every push, Python 3.11, ubuntu, install, run pytest -m "not slow". Commit. Stop with a CHECKPOINT report.

**CHECKPOINT 2:** read each lane's report and diff. Make sure you can explain every tool docstring and every edge in the graph. Merge the three branches into main, run `pytest -m "not slow"` locally, push, and check the Actions run is green.

### Phase 3: integration and eval (one session)

Only after Phase 1 is frozen and Phase 2 is merged.

> Read PLAN.md, CLAUDE.md, DECISIONS.md and eval/gold/FROZEN.md. Check current state and that the sha256 values in FROZEN.md still match the gold files; stop if they don't. Replace any stubs with Lane A's real tools, put Shrinidhi's relaxation policy (from DECISIONS.md) into the system prompt, and write tests/test_integration.py (one real Groq call on one dev task, marked slow). Run the dev split. You may adjust the system prompt and tool docstrings based on dev failures ONLY; log every change in DECISIONS.md with the dev failure that motivated it. When the dev split looks reasonable, freeze the prompt (commit), then run the test split EXACTLY ONCE. Do not change anything after seeing test results. Write eval/results/SUMMARY.md with counts per metric, every failure listed with its trace, and token and latency totals. Commit. Stop with a CHECKPOINT report.

**CHECKPOINT 3:** read every test failure yourself. Check nothing was changed after the test run (git log).

### Phase 4: README and honesty pass (you, then with Claude in chat)

README covers: what it does, how to run it, the manual-first method, results as counts including every failure and false fit, token and latency cost, and limitations (single manufacturer, 2020 public listings not live stock, 15 test tasks written by one person acting as a broker, no funders or integrations, a demo of the approach, not a product). Then paste the README into chat for an over-claim audit before you show it to anyone.

### Stretch (only after Phase 4)

Langfuse tracing on the eval run. Then, if time: an MCP server exposing the two data tools.

---

## 6. Standing rules (copy into CLAUDE.md)

- Check the current repo state before writing any file. Report what you found.
- Stay inside the files your phase or lane owns.
- Never create, edit or delete anything under eval/gold/. Never read eval/gold/test.jsonl except when Phase 3 runs the test split.
- Never tune prompts or docstrings using test-split results.
- Never commit .env, API keys, or data/raw/audi.csv unless DECISIONS.md records that the licence allows it.
- No paid APIs. Groq and NVIDIA NIM only.
- Unit tests never hit the network or an LLM. Real-LLM tests are marked slow.
- Log every non-obvious choice in DECISIONS.md: what, why, what was rejected.
- Report results as counts. Don't describe results as good, strong or accurate.
- Don't push. Shrinidhi pushes after reviewing.
- Stop at the CHECKPOINT and wait.

Add to `.claude/settings.json` in the repo as a hard guarantee (deny rules apply in every mode):

```json
{
  "permissions": {
    "deny": [
      "Edit(eval/gold/**)",
      "Write(eval/gold/**)",
      "Read(eval/gold/test.jsonl)",
      "Read(.env)"
    ]
  }
}
```

Phase 3 needs the test file: the harness reads it from Python, which these rules don't block, so this only stops Claude Code opening it directly.
