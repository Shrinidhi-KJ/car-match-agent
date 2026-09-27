"""Deterministic scoring functions for PLAN.md section 4. Pure: no I/O, no LLM, no data files.

A "run" is the per-task record written by run_eval.py:
  outcome, listing_ids, relaxed_constraint, trace (list of {name, args, status, result}),
  returned_listings (listing dicts, or None for an unknown ID), plus cost fields.
"""

from typing import Any

from pydantic import ValidationError

from carmatch.checks import UNKNOWN_LISTING, violations
from carmatch.schemas import Constraints

METRICS = ["outcome_correct", "tools_correct", "constraints_correct", "relaxed_correct"]


def tools_called(trace: list[dict]) -> set[str]:
    return {step["name"] for step in trace}


def first_search_args(trace: list[dict]) -> dict | None:
    """Arguments of the first search_listings call, whether or not it succeeded."""
    return next((step["args"] for step in trace if step["name"] == "search_listings"), None)


def parse_constraints(args: dict | None) -> Constraints | None:
    """Normalised Constraints, or None if absent or invalid."""
    if args is None:
        return None
    try:
        return Constraints(**args)
    except ValidationError:
        return None


def constraints_match(actual_args: dict | None, expected: dict) -> bool:
    """Equal after normalisation; a missing field and an explicit null are the same."""
    actual = parse_constraints(actual_args)
    return actual is not None and actual.active() == Constraints(**expected).active()


def listing_violations(listings: list[dict | None], constraints: Constraints) -> list[list[str]]:
    return [[UNKNOWN_LISTING] if lst is None else violations(lst, constraints) for lst in listings]


def score_task(task: dict, run: dict) -> dict[str, Any]:
    """Score one task. Metrics that don't apply to the task are None (excluded from counts)."""
    trace = run.get("trace", [])
    outcome = run.get("outcome")
    expected_outcome = task["expected_outcome"]
    expected_constraints = task.get("expected_constraints")

    search_expected = "search_listings" in task["expected_tools"] and expected_constraints is not None
    constraints_correct = (
        constraints_match(first_search_args(trace), expected_constraints) if search_expected else None
    )

    relaxed_correct = None
    if expected_outcome == "relaxed_match":
        relaxed_correct = outcome == "relaxed_match" and run.get("relaxed_constraint") == task.get("expected_relaxed")

    # False fit is judged against the customer's real constraints (the gold ones). For tasks with
    # no gold constraints, fall back to what the agent itself searched for.
    judge = (
        Constraints(**expected_constraints)
        if expected_constraints is not None
        else parse_constraints(first_search_args(trace))
    )
    returned = run.get("returned_listings", [])
    per_listing = listing_violations(returned, judge) if judge is not None else []
    false_fit = outcome == "match" and any(per_listing)

    # Logged, not a headline metric: a relaxed_match whose listings break constraints other than the named one.
    relaxed_extra = False
    if outcome == "relaxed_match" and judge is not None:
        named = run.get("relaxed_constraint")
        relaxed_extra = any(set(v) - {named} for v in per_listing)

    return {
        "id": task["id"],
        "expected_outcome": expected_outcome,
        "outcome": outcome,
        "outcome_correct": outcome == expected_outcome,  # gave_up / error (None) never equal a gold outcome
        "tools_correct": tools_called(trace) == set(task["expected_tools"]),
        "tools_called": sorted(tools_called(trace)),
        "constraints_correct": constraints_correct,
        "relaxed_correct": relaxed_correct,
        "false_fit": false_fit,
        "false_fit_details": {
            lid: v for lid, v in zip(run.get("listing_ids", []), per_listing) if v
        } if false_fit else {},
        "relaxed_extra_violation": relaxed_extra,
        "gave_up": outcome == "gave_up",
        "error": run.get("error"),
        "llm_calls": run.get("llm_calls", 0),
        "input_tokens": run.get("input_tokens", 0),
        "output_tokens": run.get("output_tokens", 0),
        "latency_s": run.get("latency_s", 0.0),
        "rate_limit_retries": run.get("retries", 0),
        "tool_use_failed_retries": run.get("tool_use_failed_retries", 0),
    }


def summarize(scores: list[dict]) -> dict[str, Any]:
    """Counts only: each metric is {"correct": k, "of": n} over the tasks it applies to."""
    summary: dict[str, Any] = {"tasks": len(scores)}
    for metric in METRICS:
        applicable = [s[metric] for s in scores if s[metric] is not None]
        summary[metric] = {"correct": sum(applicable), "of": len(applicable)}
    for flag in ["false_fit", "relaxed_extra_violation", "gave_up"]:
        summary[flag] = sum(1 for s in scores if s[flag])
    summary["errors"] = sum(1 for s in scores if s["error"])
    summary["llm_calls"] = sum(s["llm_calls"] for s in scores)
    summary["input_tokens"] = sum(s["input_tokens"] for s in scores)
    summary["output_tokens"] = sum(s["output_tokens"] for s in scores)
    summary["latency_s"] = round(sum(s["latency_s"] for s in scores), 2)
    summary["rate_limit_retries"] = sum(s.get("rate_limit_retries", 0) for s in scores)
    summary["tool_use_failed_retries"] = sum(s.get("tool_use_failed_retries", 0) for s in scores)
    return summary


def _yn(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "no")


def summary_markdown(summary: dict, scores: list[dict], title: str) -> str:
    n = summary["tasks"]
    lines = [f"# {title}", "", "| metric | count |", "|---|---|"]
    for metric in METRICS:
        m = summary[metric]
        lines.append(f"| {metric.replace('_', ' ')} | {m['correct']} of {m['of']} |")
    lines += [
        f"| false fits | {summary['false_fit']} of {n} |",
        f"| relaxed answers breaking another constraint | {summary['relaxed_extra_violation']} of {n} |",
        f"| gave up (step cap) | {summary['gave_up']} of {n} |",
        f"| errors | {summary['errors']} of {n} |",
        f"| LLM calls | {summary['llm_calls']} |",
        f"| input tokens | {summary['input_tokens']} |",
        f"| output tokens | {summary['output_tokens']} |",
        f"| wall-clock seconds (incl. pacing waits) | {summary['latency_s']} |",
        f"| HTTP 429 retries | {summary['rate_limit_retries']} |",
        f"| malformed tool call retries (Groq tool_use_failed) | {summary['tool_use_failed_retries']} |",
        "",
        "| task | expected | got | tools | constraints | relaxed | false fit | calls | tokens in/out | s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scores:
        lines.append(
            f"| {s['id']} | {s['expected_outcome']} | {s['outcome'] or 'error'} | {_yn(s['tools_correct'])} "
            f"| {_yn(s['constraints_correct'])} | {_yn(s['relaxed_correct'])} | {_yn(s['false_fit'])} "
            f"| {s['llm_calls']} | {s['input_tokens']}/{s['output_tokens']} | {s['latency_s']} |"
        )
    return "\n".join(lines) + "\n"
