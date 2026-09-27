"""Tests for eval/run_eval.py: pacing, 429 retries, gold guards, and an end-to-end run with a fake graph."""

import hashlib
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from carmatch.graph import build_graph
from eval.run_eval import (
    EvalCaller,
    EventLog,
    TokenPacer,
    check_frozen,
    check_test_run_allowed,
    load_tasks,
    run_tasks,
    write_results,
)
from fakes import call, scripted


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_pacer_does_not_wait_under_budget():
    clock = FakeClock()
    pacer = TokenPacer(1000, clock=clock, sleep=clock.sleep)
    pacer.record(400)
    assert pacer.wait(500) == 0
    assert clock.sleeps == []


def test_pacer_waits_for_oldest_usage_to_leave_window():
    clock = FakeClock()
    pacer = TokenPacer(1000, clock=clock, sleep=clock.sleep)
    pacer.record(700)
    clock.now = 10
    pacer.record(200)
    waited = pacer.wait(500)  # 900 + 500 > 1000 until the 700 expires at t=60
    assert clock.now >= 60
    assert waited == pytest.approx(50.05)
    assert pacer.used() == 200


def test_pacer_never_blocks_forever_on_an_oversized_first_call():
    clock = FakeClock()
    assert TokenPacer(100, clock=clock, sleep=clock.sleep).wait(5000) == 0


class RateLimited(Exception):
    status_code = 429


class Flaky:
    """Raises 429 `failures` times, then replies."""

    def __init__(self, failures, exc=RateLimited):
        self.failures, self.exc, self.calls = failures, exc, 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc("rate limited")
        return AIMessage(content="ok", usage_metadata={"input_tokens": 50, "output_tokens": 5, "total_tokens": 55})


def test_caller_retries_429_with_backoff_and_logs(tmp_path):
    clock = FakeClock()
    log = EventLog(tmp_path / "log.jsonl")
    caller = EvalCaller(TokenPacer(10_000, clock=clock, sleep=clock.sleep), log, sleep=clock.sleep, clock=clock)
    caller.start_task("t1")
    reply = caller(Flaky(2), [HumanMessage("hi")])
    assert reply.content == "ok"
    assert clock.sleeps == [2, 4]
    assert caller.stats["retries"] == 2 and caller.stats["llm_calls"] == 1 and caller.stats["input_tokens"] == 50
    events = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["llm_retry_429", "llm_retry_429", "llm_call"]
    assert events[-1]["task_id"] == "t1"


def test_caller_gives_up_after_max_retries_and_does_not_retry_other_errors(tmp_path):
    clock = FakeClock()
    caller = EvalCaller(TokenPacer(10_000, clock=clock, sleep=clock.sleep), EventLog(None),
                        sleep=clock.sleep, clock=clock, max_retries=2)
    with pytest.raises(RateLimited):
        caller(Flaky(10), [HumanMessage("hi")])
    other = Flaky(1, exc=ValueError)
    with pytest.raises(ValueError):
        caller(other, [HumanMessage("hi")])
    assert other.calls == 1


def _gold_dir(tmp_path, content='{"id": "d1"}\n'):
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "dev.jsonl").write_text(content)
    return gold


def test_frozen_check_accepts_matching_hash(tmp_path):
    gold = _gold_dir(tmp_path)
    digest = hashlib.sha256((gold / "dev.jsonl").read_bytes()).hexdigest()
    (gold / "FROZEN.md").write_text(f"- dev.jsonl sha256: {digest}\n")
    assert check_frozen(gold, "dev") == gold / "dev.jsonl"


def test_frozen_check_rejects_changed_file_or_missing_freeze(tmp_path):
    gold = _gold_dir(tmp_path)
    with pytest.raises(SystemExit, match="not frozen"):
        check_frozen(gold, "dev")
    (gold / "FROZEN.md").write_text(f"- dev.jsonl sha256: {'0' * 64}\n")
    with pytest.raises(SystemExit, match="does not match"):
        check_frozen(gold, "dev")


def test_test_split_needs_confirmation_and_runs_once(tmp_path):
    with pytest.raises(SystemExit, match="confirm"):
        check_test_run_allowed(tmp_path, confirmed=False)
    check_test_run_allowed(tmp_path, confirmed=True)
    (tmp_path / "test_20260101-000000.json").write_text("{}")
    with pytest.raises(SystemExit, match="only be run once"):
        check_test_run_allowed(tmp_path, confirmed=True)


TASKS = [
    {"id": "f1", "request": "A3 automatic 2018+ under 40k miles max 18000",
     "expected_constraints": {"model": "A3", "year_min": 2018, "transmission": "Automatic",
                              "mileage_max": 40000, "price_max": 18000},
     "expected_outcome": "match", "expected_relaxed": None,
     "expected_tools": ["search_listings", "submit_answer"], "broker_notes": "invented for a unit test"},
    {"id": "f2", "request": "something nice", "expected_constraints": None,
     "expected_outcome": "ask_customer", "expected_relaxed": None,
     "expected_tools": ["ask_customer"], "broker_notes": "invented for a unit test"},
]


def test_end_to_end_with_fake_graph(tmp_path, mini_df):
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("\n".join(json.dumps(t) for t in TASKS) + "\n")
    tasks = load_tasks(tasks_file)

    script = [
        call("search_listings", TASKS[0]["expected_constraints"], tokens=(1500, 80)),
        call("submit_answer", {"outcome": "match", "listing_ids": ["L00001", "L00008"], "message": "x"}, tokens=(1800, 60)),
        call("ask_customer", {"question": "Which model?"}, tokens=(1400, 30)),
    ]
    clock = FakeClock()
    log = EventLog(tmp_path / "run.log.jsonl")
    caller = EvalCaller(TokenPacer(10_000, clock=clock, sleep=clock.sleep), log, sleep=clock.sleep, clock=clock)
    graph = build_graph(scripted(script), listings=mini_df, llm_caller=caller)

    records = run_tasks(tasks, graph, caller, log, clock=clock)
    meta = {"name": "fake", "timestamp": "t", "model": "fake", "reasoning_effort": "low",
            "prompt_fingerprint": graph.prompt_fingerprint, "git": {"commit": "abc1234", "dirty": False}}
    json_path, md_path = write_results(records, meta, tmp_path, "fake_t")

    first, second = records
    assert first["run"]["outcome"] == "match"
    assert [s["name"] for s in first["run"]["trace"]] == ["search_listings", "submit_answer"]
    assert first["run"]["input_tokens"] == 3300 and first["run"]["output_tokens"] == 140
    assert first["run"]["llm_calls"] == 2
    assert first["scores"]["false_fit"] is True  # L00008 is 18001 GBP
    assert first["run"]["in_graph_false_fit"] is True
    assert second["run"]["outcome"] == "ask_customer" and second["scores"]["outcome_correct"]

    saved = json.loads(json_path.read_text())
    assert saved["summary"]["outcome_correct"] == {"correct": 2, "of": 2}
    assert saved["summary"]["false_fit"] == 1
    assert "| outcome correct | 2 of 2 |" in md_path.read_text()

    events = [json.loads(line) for line in (tmp_path / "run.log.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds.count("llm_call") == 3 and kinds.count("tool_call") == 3


def test_graph_exception_is_recorded_not_raised(tmp_path, mini_df):
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(json.dumps(TASKS[1]) + "\n")
    caller = EvalCaller(TokenPacer(10_000), EventLog(None), max_retries=0)
    graph = build_graph(scripted([]), listings=mini_df, llm_caller=caller)  # model has nothing to say: raises
    [record] = run_tasks(load_tasks(tasks_file), graph, caller, EventLog(None))
    assert record["run"]["error"]
    assert record["scores"]["outcome_correct"] is False
