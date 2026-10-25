"""即時辨識的子行程：音訊從佇列進來，草稿和確定的句子從佇列出去。

放在子行程裡：模型當掉不會帶走介面；停止時結束行程，顯示記憶體完整釋放。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import traceback

import numpy as np

from omni2.workers.winjob import kill_with_parent

log = logging.getLogger(__name__)

# 佇列裡積壓超過這麼多塊（每塊約 0.1 秒）就不做草稿，先追上進度
_BEHIND_CHUNKS = 5


def _make_recognizers(cfg: dict, emit) -> dict:
    from omni2.config.settings import SttSettings
    from omni2.live.recognizers import FunAsrRealtimeRecognizer, LocalRecognizer
    from omni2.stt.registry import create_engine
    from omni2.vad.presets import get_preset

    sources = cfg["sources"]
    if cfg["engine"] == "funasr":
        return {s: FunAsrRealtimeRecognizer(s, cfg["funasr_model"], cfg["region"], cfg["language"]) for s in sources}

    stt = SttSettings(**cfg["stt"])
    emit({"type": "status", "message": f"載入語音辨識模型 {stt.active_model()}"})
    engine = create_engine(stt, live=True)  # 多個來源共用同一個模型
    engine.load()
    engine.transcribe_text(np.zeros(16000, np.float32), cfg["language"])  # 暖機：第一次推論特別慢
    preset = get_preset(cfg["vad_preset"])
    return {s: LocalRecognizer(engine, preset, s, cfg["language"], cfg["partial_ms"], cfg["max_ms"])
            for s in sources}


def _main(cfg: dict, inq, outq) -> None:
    import os
    import sys

    for name in ("stdout", "stderr"):  # 從 GUI 啟動時沒有主控台
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    emit = outq.put
    try:
        recognizers = _make_recognizers(cfg, emit)
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "fatal": True, "message": f"{type(e).__name__}: {e}",
              "traceback": traceback.format_exc()})
        return
    emit({"type": "ready"})
    while True:
        msg = inq.get()
        if msg is None:
            break
        source, audio = msg
        try:
            for event in recognizers[source].feed(audio, behind=inq.qsize() > _BEHIND_CHUNKS):
                emit(event)
        except Exception as e:  # noqa: BLE001 — 一塊音訊出錯不要讓整個即時翻譯停掉
            emit({"type": "error", "message": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
    for recognizer in recognizers.values():
        for event in recognizer.flush():
            emit(event)
    emit({"type": "eof"})  # 最後的結果都送完了


class LiveSttProcess:
    def __init__(self, cfg: dict):
        ctx = mp.get_context("spawn")
        self.inq = ctx.Queue()
        self.outq = ctx.Queue()
        self.proc = ctx.Process(target=_main, args=(cfg, self.inq, self.outq), daemon=True, name="live-stt")

    def start(self) -> None:
        self.proc.start()
        kill_with_parent(self.proc.sentinel)

    def send(self, source: str, audio: np.ndarray) -> None:
        self.inq.put((source, audio))

    def get(self, timeout: float = 0.1) -> dict | None:
        try:
            return self.outq.get(timeout=timeout)
        except queue.Empty:
            if not self.proc.is_alive():
                return {"type": "error", "fatal": True,
                        "message": f"語音辨識行程意外結束（exit code {self.proc.exitcode}）"}
            return None

    def stop(self, timeout: float = 10) -> list[dict]:
        """請子行程把講到一半的句子處理完再結束，回傳最後的事件。"""
        self.inq.put(None)
        events = []
        while self.proc.is_alive() or not self.outq.empty():
            try:
                event = self.outq.get(timeout=timeout)
            except queue.Empty:
                break
            if event.get("type") == "eof":
                break
            events.append(event)
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.terminate()
        return events
