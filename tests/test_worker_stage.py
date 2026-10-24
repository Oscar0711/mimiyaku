import os
import threading
import time

import pytest

from omni2.workers.stage import Cancelled, WorkerCrashed, WorkerError, run_stage


# 以下函式會在 spawn 出來的子行程裡執行
def echo_worker(payload, emit):
    import logging

    logging.getLogger("child").info("child log")
    emit({"type": "hello", "n": payload["n"]})
    return {"double": payload["n"] * 2, "pid": os.getpid()}


def failing_worker(payload, emit):
    raise ValueError("boom")


def crashing_worker(payload, emit):
    os._exit(3)


def sleeping_worker(payload, emit):
    emit({"type": "started"})
    time.sleep(60)


def test_result_and_events():
    events = []
    result = run_stage("test_worker_stage:echo_worker", {"n": 21}, events.append)
    assert result["double"] == 42
    assert result["pid"] != os.getpid()
    assert {"type": "hello", "n": 21} in events
    assert any(e.get("type") == "log" and e["message"] == "child log" for e in events)


def test_error_is_propagated():
    with pytest.raises(WorkerError, match="ValueError: boom") as info:
        run_stage("test_worker_stage:failing_worker", {}, lambda e: None)
    assert "Traceback" in info.value.detail


def test_crash_is_detected():
    with pytest.raises(WorkerCrashed, match="exit code 3"):
        run_stage("test_worker_stage:crashing_worker", {}, lambda e: None)


def test_cancel_terminates_worker():
    cancel = threading.Event()

    def on_event(ev):
        if ev.get("type") == "started":
            cancel.set()

    start = time.monotonic()
    with pytest.raises(Cancelled):
        run_stage("test_worker_stage:sleeping_worker", {}, on_event, cancel)
    assert time.monotonic() - start < 20
