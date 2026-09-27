"""Count clean vs malformed terminal calls across diagnosis runs (smoke data only).

    python -m eval.diagnose_terminal_calls eval/results/smoke_A-*.json eval/results/smoke_B-*.json ...

For each results file (and its .log.jsonl): tasks run, tasks ending in a clean terminal call,
malformed calls rejected by the provider (tool_use_failed, every attempt counted), terminal calls
our own schema rejected, gave_up, errors, tokens (successful calls), 429 retries, wall-clock.
"""

import json
import sys
from pathlib import Path

TERMINAL_OUTCOMES = {"match", "relaxed_match", "no_match", "ask_customer"}


def summarise(results_path: Path) -> dict:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    log_path = results_path.with_suffix(".log.jsonl")
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    runs = [t["run"] for t in data["tasks"]]
    # Provider-rejected malformed output: tool_use_failed (retried by the harness) and
    # output_parse_failed (not retried). Every attempt is counted.
    malformed = [
        e for e in events
        if e["event"] == "llm_retry_tool_use_failed"
        or (e["event"] == "llm_error" and e.get("code") in ("tool_use_failed", "output_parse_failed"))
    ]
    schema_rejected = sum(
        1 for r in runs for step in r["trace"]
        if step["name"] in ("submit_answer", "ask_customer") and step["status"] == "error"
    )
    return {
        "file": results_path.name,
        "model": data["meta"]["model"],
        "effort": data["meta"]["reasoning_effort"],
        "tasks": len(runs),
        "clean_terminal": sum(1 for r in runs if r["outcome"] in TERMINAL_OUTCOMES),
        "malformed_calls": len(malformed),
        "malformed_by_code": {c: sum(1 for e in malformed if e.get("code", "tool_use_failed") == c)
                              for c in sorted({e.get("code") or "tool_use_failed" for e in malformed})},
        "tasks_with_malformed": len({e["task_id"] for e in malformed}),
        "schema_rejected_terminal": schema_rejected,
        "gave_up": sum(1 for r in runs if r["outcome"] == "gave_up"),
        "errors": sum(1 for r in runs if r["error"]),
        "tokens_in": data["summary"]["input_tokens"],
        "tokens_out": data["summary"]["output_tokens"],
        "retries_429": data["summary"].get("rate_limit_retries", 0),
        "wall_s": data["summary"]["latency_s"],
        "outcomes": {t["task"]["id"]: t["run"]["outcome"] or "error" for t in data["tasks"]},
        "malformed_examples": [
            {"task_id": e["task_id"], "code": e.get("code"), "message": e.get("message"),
             "failed_generation": e.get("failed_generation")}
            for e in malformed
        ],
    }


if __name__ == "__main__":
    rows = [summarise(Path(p)) for p in sys.argv[1:]]
    print(json.dumps(rows, indent=2, ensure_ascii=True))
