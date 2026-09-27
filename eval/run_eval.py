"""Run the agent over a task file and write results.

    python -m eval.run_eval --tasks eval/smoke/smoke.jsonl      # any task file (e.g. the smoke set)
    python -m eval.run_eval --split dev                          # eval/gold/dev.jsonl, frozen hashes checked
    python -m eval.run_eval --split test --confirm-test-split    # the one-time test run

Writes eval/results/<name>_<timestamp>.json (per-task trace, tokens, latency, scores, summary),
<name>_<timestamp>.md (count tables) and <name>_<timestamp>.log.jsonl (one JSON line per LLM call
and per tool call).
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from carmatch.graph import initial_state
from carmatch.schemas import GoldTask
from carmatch.tools import listing_record
from eval.score import score_task, summarize, summary_markdown

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "eval" / "gold"
RESULTS_DIR = ROOT / "eval" / "results"

TOKENS_PER_MINUTE = 8000
PACING_BUDGET = 7200  # stay 10% under the Groq free-tier limit
TOOL_SCHEMA_OVERHEAD = 1200  # rough prompt tokens for system prompt + four tool schemas
EXPECTED_OUTPUT = 400
MAX_429_RETRIES = 6
MAX_TOOL_USE_FAILED_RETRIES = 2


# ---------------------------------------------------------------- logging


class EventLog:
    """Append-only JSON-lines log. One object per line, each with a UTC timestamp and an event name."""

    def __init__(self, path: Path | None):
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        if self.path is None:
            return
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------- pacing and retries


class TokenPacer:
    """Sliding 60-second window: wait before a call if recent usage plus the estimate would exceed the budget."""

    def __init__(
        self,
        budget: int = PACING_BUDGET,
        *,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.budget, self.window_s, self.clock, self.sleep = budget, window_s, clock, sleep
        self.events: deque[tuple[float, int]] = deque()

    def _prune(self) -> None:
        now = self.clock()
        while self.events and self.events[0][0] <= now - self.window_s:
            self.events.popleft()

    def used(self) -> int:
        self._prune()
        return sum(tokens for _, tokens in self.events)

    def wait(self, estimate: int) -> float:
        """Block until the estimate fits in the window. Returns seconds waited."""
        waited = 0.0
        while self.events and self.used() + estimate > self.budget:
            delay = max(self.events[0][0] + self.window_s - self.clock(), 0.0) + 0.05
            self.sleep(delay)
            waited += delay
        return waited

    def record(self, tokens: int) -> None:
        self.events.append((self.clock(), tokens))


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Rough prompt size (4 chars per token) plus tool-schema overhead and expected output."""
    chars = sum(len(str(m.content)) + len(json.dumps(getattr(m, "tool_calls", []) or [])) for m in messages)
    return chars // 4 + TOOL_SCHEMA_OVERHEAD + EXPECTED_OUTPUT


def is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or type(exc).__name__ == "RateLimitError"


def is_tool_use_failed(exc: Exception) -> bool:
    """Groq 400 when the model's tool call can't be parsed (e.g. gpt-oss naming a tool "json")."""
    return getattr(exc, "status_code", None) == 400 and "tool_use_failed" in str(exc)


def retry_after_s(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    value = getattr(response, "headers", {}).get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class EvalCaller:
    """The graph's llm_caller for eval runs: paces, retries HTTP 429 with backoff, logs, and counts per task."""

    def __init__(
        self,
        pacer: TokenPacer,
        log: EventLog,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        max_retries: int = MAX_429_RETRIES,
        max_tool_use_failed_retries: int = MAX_TOOL_USE_FAILED_RETRIES,
    ):
        self.pacer, self.log, self.sleep, self.clock, self.max_retries = pacer, log, sleep, clock, max_retries
        self.max_tool_use_failed_retries = max_tool_use_failed_retries
        self.start_task("")

    def start_task(self, task_id: str) -> None:
        self.task_id = task_id
        self.stats = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "retries": 0,
                      "tool_use_failed_retries": 0, "waited_s": 0.0, "llm_s": 0.0}

    def __call__(self, runnable, messages: list[BaseMessage]) -> AIMessage:
        estimate = estimate_tokens(messages)
        waited = self.pacer.wait(estimate)
        self.stats["waited_s"] += waited
        rate_limited = tool_failed = 0
        while True:
            t0 = self.clock()
            try:
                reply = runnable.invoke(messages)
            except Exception as exc:
                if is_rate_limit(exc) and rate_limited < self.max_retries:
                    backoff = retry_after_s(exc) or min(2 ** rate_limited * 2, 60)
                    rate_limited += 1
                    self.stats["retries"] += 1
                    self.log.write("llm_retry_429", task_id=self.task_id, attempt=rate_limited, backoff_s=backoff)
                    self.sleep(backoff)
                    continue
                if is_tool_use_failed(exc) and tool_failed < self.max_tool_use_failed_retries:
                    tool_failed += 1
                    self.stats["tool_use_failed_retries"] += 1
                    self.pacer.record(estimate)  # the failed generation still used tokens
                    self.log.write("llm_retry_tool_use_failed", task_id=self.task_id, attempt=tool_failed,
                                   error=repr(exc)[:500])
                    continue
                self.log.write("llm_error", task_id=self.task_id, error=repr(exc)[:500],
                               rate_limit_retries=rate_limited, tool_use_failed_retries=tool_failed)
                raise
            latency = self.clock() - t0
            usage = reply.usage_metadata or {}
            tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            self.pacer.record(tokens_in + tokens_out or estimate)
            self.stats["llm_calls"] += 1
            self.stats["input_tokens"] += tokens_in
            self.stats["output_tokens"] += tokens_out
            self.stats["llm_s"] += latency
            self.log.write(
                "llm_call",
                task_id=self.task_id,
                call=self.stats["llm_calls"],
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                reasoning_tokens=(usage.get("output_token_details") or {}).get("reasoning"),
                estimate=estimate,
                latency_s=round(latency, 3),
                waited_s=round(waited, 3),
                retries=rate_limited,
                tool_use_failed_retries=tool_failed,
                tool_calls=[tc["name"] for tc in reply.tool_calls],
            )
            return reply


# ---------------------------------------------------------------- gold-set guards


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen(gold_dir: Path, split: str) -> Path:
    """The split file must exist and its sha256 must appear on a FROZEN.md line naming it."""
    path = gold_dir / f"{split}.jsonl"
    frozen = gold_dir / "FROZEN.md"
    if not frozen.exists():
        raise SystemExit(f"{frozen} missing: the gold set is not frozen, refusing to run.")
    if not path.exists():
        raise SystemExit(f"{path} missing.")
    recorded = [
        h.lower()
        for line in frozen.read_text(encoding="utf-8").splitlines()
        if f"{split}.jsonl" in line
        for h in re.findall(r"\b[0-9a-fA-F]{64}\b", line)
    ]
    actual = sha256_file(path)
    if actual not in recorded:
        raise SystemExit(f"sha256 of {path.name} ({actual}) does not match FROZEN.md {recorded}; refusing to run.")
    return path


def check_test_run_allowed(results_dir: Path, confirmed: bool) -> None:
    if not confirmed:
        raise SystemExit("The test split runs exactly once. Re-run with --confirm-test-split to do it.")
    previous = sorted(results_dir.glob("test_*.json"))
    if previous:
        raise SystemExit(f"A test-split result already exists ({previous[0].name}); it may only be run once.")


# ---------------------------------------------------------------- running


def load_tasks(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [GoldTask.model_validate_json(ln).model_dump() for ln in lines]


def build_trace(messages: list[BaseMessage]) -> list[dict]:
    """Every tool call in order, with its arguments, status and result."""
    calls: dict[str, dict] = {}
    trace = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                step = {"name": tc["name"], "args": tc["args"], "status": None, "result": None}
                calls[tc["id"]] = step
                trace.append(step)
        elif isinstance(m, ToolMessage) and m.tool_call_id in calls:
            calls[m.tool_call_id].update(status=m.status, result=m.content)
    return trace


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def run_tasks(tasks: list[dict], graph, caller: EvalCaller, log: EventLog, *, clock=time.perf_counter) -> list[dict]:
    """Run the graph on each task; return one record per task (run fields + scores)."""
    records = []
    for task in tasks:
        caller.start_task(task["id"])
        log.write("task_start", task_id=task["id"])
        t0 = clock()
        state, error = None, None
        try:
            # stream() rather than invoke() so the last good state (and its trace) survives an exception.
            for state in graph.stream(initial_state(task["request"]), stream_mode="values"):
                pass
            if state is not None and state.get("outcome") is None:
                error = "graph ended without an outcome"
        except Exception as exc:  # recorded as a failed task, not a crashed eval
            error = repr(exc)[:500]
        latency = clock() - t0

        messages = state["messages"] if state else []
        trace = build_trace(messages)
        for i, step in enumerate(trace):
            log.write("tool_call", task_id=task["id"], index=i, **step)
        ids = (state or {}).get("listing_ids") or []
        run = {
            "outcome": (state or {}).get("outcome"),
            "listing_ids": ids,
            "relaxed_constraint": (state or {}).get("relaxed_constraint"),
            "final_message": (state or {}).get("final_message"),
            "in_graph_false_fit": (state or {}).get("false_fit"),
            "returned_listings": [
                listing_record(lid, graph.listings.loc[lid]) if lid in graph.listings.index else None for lid in ids
            ],
            "trace": trace,
            "error": error,
            "latency_s": round(latency, 3),
            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in caller.stats.items()},
        }
        scores = score_task(task, run)
        log.write("task_end", task_id=task["id"], outcome=run["outcome"], error=error, latency_s=run["latency_s"])
        records.append({"task": task, "run": run, "scores": scores})
    return records


def write_results(records: list[dict], meta: dict, out_dir: Path, stem: str) -> tuple[Path, Path]:
    scores = [r["scores"] for r in records]
    summary = summarize(scores)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps({"meta": meta, "summary": summary, "tasks": records}, indent=2, default=str), encoding="utf-8")
    title = f"{meta['name']} run {meta['timestamp']}"
    header = (
        f"Model `{meta['model']}` (reasoning_effort {meta['reasoning_effort']}), commit `{meta['git']['commit'][:7]}`"
        f"{' (dirty)' if meta['git']['dirty'] else ''}, prompt fingerprint `{meta['prompt_fingerprint'][:12]}`.\n\n"
    )
    md = summary_markdown(summary, scores, title)
    md_path.write_text(md.replace("\n\n", "\n\n" + header, 1), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split", choices=["dev", "test"])
    source.add_argument("--tasks", type=Path, help="any task file in gold-set format (not under eval/gold/)")
    parser.add_argument("--confirm-test-split", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--budget", type=int, default=PACING_BUDGET, help="tokens per rolling minute")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Windows consoles can't encode every character

    if args.split:
        if args.split == "test":
            check_test_run_allowed(args.out_dir, args.confirm_test_split)
        tasks_path, name = check_frozen(GOLD_DIR, args.split), args.split
    else:
        if GOLD_DIR in args.tasks.resolve().parents:
            raise SystemExit("Use --split for gold files, so the frozen-hash check runs.")
        tasks_path, name = args.tasks, args.tasks.stem

    from carmatch.graph import build_graph
    from carmatch.llm import GROQ_MODEL, REASONING_EFFORT, make_llm

    tasks = load_tasks(tasks_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    stem = f"{name}_{timestamp}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(args.out_dir / f"{stem}.log.jsonl")
    caller = EvalCaller(TokenPacer(args.budget), log)
    graph = build_graph(make_llm(max_retries=0), llm_caller=caller)

    meta = {
        "name": name,
        "timestamp": timestamp,
        "tasks_file": str(tasks_path.resolve().relative_to(ROOT)) if ROOT in tasks_path.resolve().parents else str(tasks_path),
        "tasks_sha256": sha256_file(tasks_path),
        "model": GROQ_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "prompt_fingerprint": graph.prompt_fingerprint,
        "pacing_budget_tokens_per_min": args.budget,
        "git": git_state(),
    }
    log.write("run_start", **meta)
    records = run_tasks(tasks, graph, caller, log)
    json_path, md_path = write_results(records, meta, args.out_dir, stem)
    log.write("run_end", results=str(json_path))
    print(md_path.read_text(encoding="utf-8"))
    print(f"wrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
