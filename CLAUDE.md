# car-match-agent: standing rules

Read PLAN.md (the single source of truth) and DECISIONS.md before doing anything.

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

## Practical notes

- Python venv: `.venv/Scripts/python.exe` (Windows). Run tests with `.venv/Scripts/python.exe -m pytest -m "not slow"`.
- Unit tests and CI use only `tests/fixtures/mini.csv`, never `data/raw/audi.csv`.
