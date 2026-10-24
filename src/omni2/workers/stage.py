"""在獨立子行程裡跑一個階段（例如 STT）。

階段結束＝子行程結束，模型佔用的 VRAM 由作業系統完整回收；子行程當掉（CUDA OOM）
也不會連帶讓主程式當掉。子行程透過 emit() 回報進度，大的結果寫成檔案，不走佇列。
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import queue
import threading
import traceback
from typing import Any, Callable

from omni2.workers.winjob import kill_with_parent

EventFn = Callable[[dict], None]


class WorkerError(RuntimeError):
    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


class WorkerCrashed(WorkerError):
    pass


class Cancelled(Exception):
    pass


class _QueueLogHandler(logging.Handler):
    def __init__(self, q):
        super().__init__(logging.INFO)
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put({"kind": "event", "data": {
                "type": "log", "level": record.levelname, "message": record.getMessage()}})
        except Exception:
            pass


def _entry(target: str, payload: dict, q) -> None:
    root = logging.getLogger()
    root.handlers[:] = [_QueueLogHandler(q)]
    root.setLevel(logging.INFO)

    def emit(event: dict) -> None:
        q.put({"kind": "event", "data": event})

    try:
        module, fn = target.split(":")
        result = getattr(importlib.import_module(module), fn)(payload, emit)
        q.put({"kind": "result", "data": result})
    except BaseException as e:  # noqa: BLE001 — 子行程裡的任何錯誤都要傳回主程式
        q.put({"kind": "error", "data": {"message": f"{type(e).__name__}: {e}",
                                          "traceback": traceback.format_exc()}})


def run_stage(
    target: str,
    payload: dict,
    on_event: EventFn,
    cancel: threading.Event | None = None,
) -> Any:
    """target 是 'module:function'，函式簽名為 fn(payload, emit) -> 可 pickle 的結果。"""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_entry, args=(target, payload, q), daemon=True)
    proc.start()
    kill_with_parent(proc.sentinel)
    finished = False
    try:
        while True:
            try:
                msg = q.get(timeout=0.2)
            except queue.Empty:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                if not proc.is_alive():
                    # 行程結束前放進佇列的訊息可能還沒讀完，再撈一次
                    try:
                        msg = q.get(timeout=0.5)
                    except queue.Empty:
                        raise WorkerCrashed(f"worker 意外結束（exit code {proc.exitcode}）") from None
                else:
                    continue
            kind = msg["kind"]
            if kind == "event":
                on_event(msg["data"])
            elif kind == "result":
                finished = True
                return msg["data"]
            else:
                finished = True
                raise WorkerError(msg["data"]["message"], msg["data"]["traceback"])
    finally:
        # 正常結束時給子行程一點時間收尾；取消或出錯時直接結束它
        if finished and proc.is_alive():
            proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        q.close()
