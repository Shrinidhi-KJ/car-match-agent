# car-match-agent

A small LangGraph agent that takes a plain-English used-car request ("a 2018 or newer Audi A3, automatic, under 40k miles, under £18,000") and matches it to listings from a public dataset. It either returns matching cars, returns cars with one constraint loosened, says nothing suitable exists, or asks a clarifying question.

It is a demonstration of an approach (deterministic tools, an LLM that only chooses between them, and a hand-written, frozen evaluation set), not a product.

## What it does

The LLM never filters or ranks cars itself. It chooses between four tools:

| tool | what it does |
|---|---|
| `search_listings` | Deterministic pandas filter over the listings. Returns the match count and up to 5 listings ranked by price, then mileage. |
| `market_summary` | Count, price and mileage spread for a model and year range. Used after an empty search to see which constraint is blocking. |
| `ask_customer` | Ends the run with one clarifying question. |
| `submit_answer` | Ends the run with `match`, `relaxed_match` (naming the one loosened constraint) or `no_match`. |

A run ends when a terminal tool succeeds, or after 6 LLM calls (recorded as `gave_up`).

Matching rules, shared by the search tool, the checker and the gold set:
- Numeric limits are inclusive.
- "Automatic" also matches `Semi-Auto` listings.
- "Manual" matches Manual only.

**False-fit check.** When the agent answers `match`, a separate deterministic checker verifies that every returned listing satisfies every original constraint. A violation is a *false fit*: the agent presenting a car that doesn't fit as if it does. Catching this is the main reason the project exists.

**Data:** the Audi file from the Kaggle ["100,000 UK Used Car Data set"](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) (CC0: Public Domain), committed at `data/raw/audi.csv`. It has 10,668 rows; 103 exact duplicates are dropped at load time.

**LLM:** Groq `openai/gpt-oss-120b` (`reasoning_effort="low"`, temperature 0) via `langchain-groq`. An NVIDIA NIM fallback is wired in `src/carmatch/llm.py` but untested.

## How to run

Requires Python 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

pytest -m "not slow"          # unit tests: no network, no API key, use tests/fixtures/mini.csv
```

For real LLM runs, copy `.env.example` to `.env` and set `GROQ_API_KEY` (free tier is enough). Then:

```bash
pytest -m slow                                        # one real agent run
python -m eval.run_eval --tasks eval/smoke/smoke.jsonl   # pipeline smoke check (3 invented requests)
python -m eval.run_eval --split dev                   # gold dev split (needs eval/gold/ frozen)
```

Each eval run writes three files to `eval/results/`:
- a JSON file with the full tool-call trace, tokens, latency and scores for each task;
- a Markdown table of counts;
- a JSON-lines log with one line per LLM call and per tool call.

The harness keeps under Groq's free-tier limit of 8,000 tokens per minute and retries HTTP 429s with backoff.

CI (GitHub Actions) runs `pytest -m "not slow"` on every push.

## Method: manual first

1. **The expected answers are written by hand before the agent runs on them.** One person, acting as a broker, browses the real listings and writes 20 requests with the expected constraints, outcome, relaxed constraint (if any) and tools. This includes the relaxation policy the agent must follow.
2. **The gold set is frozen.** `eval/gold/FROZEN.md` records the sha256 of each split and the commit. The harness refuses to run a split whose hash doesn't match.
3. **5 dev tasks, 15 test tasks.** The prompt and tool descriptions may be adjusted only on dev failures, and every change is logged in `DECISIONS.md` with the failure that motivated it. The prompt is then frozen, and the test split is run exactly once. The harness requires an explicit flag for the test split and refuses a second run.
4. **Deterministic scoring** (`eval/score.py`). For each task it checks:
   - whether the outcome is correct;
   - whether the set of tools called is correct;
   - whether the first search's constraints are correct;
   - whether the relaxed constraint is correct;
   - whether there was a false fit (judged against the hand-written constraints, not the agent's own reading).

   Results are reported as counts ("k of n"), with n being the tasks each metric applies to.

Every results file records the model, git commit, task file hash and a fingerprint of the prompt and tool descriptions, so changes after a run can be detected.

Design decisions, with what was rejected and why, are in `DECISIONS.md`. Build progress is in `CHECKPOINTS.md`.

## Results

TODO: pending frozen gold set evaluation.

(The smoke run in `eval/results/smoke_*` uses 3 requests invented by Claude Code to check that the pipeline works. It is not an evaluation.)

## Limitations

- **One manufacturer.** Audi only.
- **Old public listings, not live stock.** The data was scraped around 2020, with no trims, colours, locations, dealers or listing dates. Prices are 2020 asking prices.
- **Small, single-author evaluation.** The 20 gold tasks are written by one person acting as a broker. 15 test tasks can show specific failures but can't support statistical claims, and none are made.
- **The "automatic includes Semi-Auto" rule is a judgement** based on UK listing conventions. The dataset doesn't say what gearbox a "Semi-Auto" listing has.
- **Model behaviour:** in the smoke run, gpt-oss-120b on Groq sometimes emitted a malformed tool call (a tool named `json` or `commentary`) that the API rejects. The harness logs and counts these; they are not hidden.
- **No integrations.** There are no funders, brokers, CRM, UI or deployment. The NIM fallback is untested.
