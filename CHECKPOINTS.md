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
