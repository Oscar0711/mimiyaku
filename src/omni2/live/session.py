"""一次即時翻譯：音訊來源 → 辨識子行程（草稿／確定）→ 串流翻譯 → 事件送給介面，並寫紀錄。

這一層不依賴 Qt，介面只要接 on_event。事件：
* {"type": "status", "message"}、{"type": "ready"}、{"type": "error", "message", "fatal"}
* {"type": "partial"|"final", "id", "src", "start", "end", "text", "lang"}（final 另有 lat_text）
* {"type": "translation", "id", "text", "done"}（第一次另有 lat_first）
* {"type": "level", "src", "rms"}：音量，給介面顯示有沒有收到聲音
* {"type": "stopped", "log": 紀錄檔路徑}
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from omni2.audio.sources import AudioApp, Device, FileSource, LoopbackSource, MicSource, ProcessSource
from omni2.config.settings import Settings
from omni2.live.record import SessionLog
from omni2.live.stt_process import LiveSttProcess
from omni2.live.translator import LiveTranslator
from omni2.translate.client import ChatClient
from omni2.translate.glossary import Glossary
from omni2.translate.server import LlamaServer

log = logging.getLogger(__name__)
EventFn = Callable[[dict], None]


@dataclass
class SourceSpec:
    kind: str  # mic | system | app | file（測試用：以真實速度播放音檔）
    label: str  # 子行程和紀錄裡用的代號
    display: str  # 給人看的名稱（我／對方）
    device: Device | None = None
    app: AudioApp | None = None
    path: str | None = None

    def open(self):
        if self.kind == "mic":
            return MicSource(self.device)
        if self.kind == "system":
            return LoopbackSource(self.device)
        if self.kind == "file":
            return FileSource(self.path)
        return ProcessSource(self.app)


class LiveSession:
    def __init__(self, settings: Settings, sources: list[SourceSpec], glossary: Glossary, on_event: EventFn):
        self.settings = settings
        self.sources = sources
        self.glossary = glossary
        self.on_event = on_event
        self.stt: LiveSttProcess | None = None
        self.server: LlamaServer | None = None
        self.client: ChatClient | None = None
        self.translator: LiveTranslator | None = None
        self.log: SessionLog | None = None
        self._open: list = []
        self._finals: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._reader: threading.Thread | None = None
        self._t0 = 0.0
        self._first_token: dict[str, float] = {}

    # ---- 啟動、停止 ----

    def start(self) -> None:
        """在背景準備（載入模型可能要半分鐘），準備好才開始收音。"""
        thread = threading.Thread(target=self._start, name="live-start", daemon=True)
        thread.start()
        self._threads.append(thread)

    def _start(self) -> None:
        live, tcfg = self.settings.live, self.settings.translate
        try:
            self.on_event({"type": "status", "message": "準備語音辨識…"})
            self.stt = LiveSttProcess(self._stt_config())
            self.stt.start()
            if live.translate and tcfg.enabled:
                self.on_event({"type": "status", "message": f"啟動翻譯模型 {Path(tcfg.model_path).name}…"})
                if tcfg.endpoint:
                    self.client = ChatClient(tcfg.endpoint, model=tcfg.endpoint_model)
                else:
                    self.server = LlamaServer(tcfg.server_exe, tcfg.model_path, ctx_size=tcfg.ctx_size,
                                              gpu_layers=tcfg.gpu_layers, extra_args=tcfg.server_args)
                    self.client = ChatClient(self.server.start())
                self.translator = LiveTranslator(self.client, tcfg.target, self._glossary_for,
                                                 context_lines=live.context_lines, opencc=tcfg.opencc)
                self.translator.warm_up()
            while not self._stop.is_set():  # 等辨識子行程載入模型
                event = self.stt.get(0.5)
                if event is None:
                    continue
                if event["type"] == "ready":
                    break
                if event["type"] == "error" and event.get("fatal"):
                    raise RuntimeError(event["message"])
                self.on_event(event)
            if self._stop.is_set():
                return
            self.log = SessionLog({"tgt_lang": tcfg.target if self.translator else None,
                                   "src_lang": live.src_lang,
                                   "labels": {s.label: s.display for s in self.sources},
                                   "sources": [s.display for s in self.sources]})
            self._t0 = time.monotonic()
            for spec in self.sources:
                source = spec.open()
                source.start(lambda audio, label=spec.label: self._on_audio(label, audio))
                self._open.append(source)
            self._reader = threading.Thread(target=self._read_stt, name="live-read", daemon=True)
            self._reader.start()
            thread = threading.Thread(target=self._translate_loop, name="live-translate", daemon=True)
            thread.start()
            self._threads.append(thread)
            self.on_event({"type": "ready"})
        except Exception as e:  # noqa: BLE001
            log.exception("即時翻譯啟動失敗")
            self.on_event({"type": "error", "fatal": True, "message": f"{type(e).__name__}: {e}"})
            self._shutdown()

    def stop(self) -> None:
        """停止收音，把講到一半的句子處理、翻譯完再結束。在背景執行，完成時送出 stopped。"""
        thread = threading.Thread(target=self._shutdown, name="live-stop", daemon=True)
        thread.start()

    def _shutdown(self) -> None:
        if self._stop.is_set() and not self._open and self.stt is None:
            return
        self._stop.set()
        for source in self._open:
            try:
                source.stop()
            except Exception as e:  # noqa: BLE001
                log.warning("停止音訊來源失敗：%s", e)
        self._open = []
        # 先讓讀取執行緒停下，子行程最後送出的結果才不會被兩邊搶著讀
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=5)
        if self.stt is not None:
            for event in self.stt.stop():
                self._handle_stt(event)
            self.stt = None
        self._finals.put(None)
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=30)
        if self.client is not None:
            self.client.close()
        if self.server is not None:
            self.server.stop()
            self.server = None
        self.on_event({"type": "stopped", "log": str(self.log.path) if self.log else None})

    # ---- 內部 ----

    def _stt_config(self) -> dict:
        live, stt = self.settings.live, self.settings.stt
        engine_settings = stt.model_copy(update={"engine": live.stt_engine, "qwen_model": live.qwen_model,
                                                 "model": live.whisper_model})
        return {"engine": live.stt_engine, "stt": engine_settings.model_dump(),
                "sources": [s.label for s in self.sources], "language": live.src_lang,
                "vad_preset": live.vad_preset, "partial_ms": live.partial_ms, "max_ms": live.max_utterance_ms,
                "funasr_model": live.funasr_model, "region": stt.cloud.dashscope_region}

    def _glossary_for(self, src_lang: str | None) -> Glossary:
        known = Glossary()
        for name in self.settings.terms.builtin:
            known = known.merged(Glossary.builtin(name, src_lang))
        return known.merged(self.glossary)

    def _on_audio(self, label: str, audio: np.ndarray) -> None:
        if self.stt is not None and not self._stop.is_set():
            self.stt.send(label, audio)
            self.on_event({"type": "level", "src": label, "rms": float(np.sqrt(np.mean(audio ** 2)))})

    def _read_stt(self) -> None:
        while not self._stop.is_set():
            event = self.stt.get(0.1) if self.stt else None
            if event is not None:
                self._handle_stt(event)

    def _handle_stt(self, event: dict) -> None:
        kind = event["type"]
        if kind == "final":
            # 從說完到原文確定所花的時間（假設擷取是即時的）
            event["lat_text"] = round(time.monotonic() - (self._t0 + event["end"]), 2)
            if not event["text"].strip():
                return
            self._finals.put(event)
        elif kind == "eof":
            return
        elif kind == "error" and event.get("fatal"):
            self.on_event(event)
            self.stop()
            return
        self.on_event(event)

    def _translate_loop(self) -> None:
        while True:
            event = self._finals.get()
            if event is None:
                break
            translation = None
            if self.translator is not None:
                def emit(e: dict, event=event) -> None:
                    if e["type"] == "first_token":
                        e = {"type": "latency", "id": e["id"],
                             "lat_first": round(e["at"] - (self._t0 + event["end"]), 2)}
                    self.on_event(e)

                try:
                    translation = self.translator.translate(event, emit)
                except Exception as e:  # noqa: BLE001 — 一句翻不出來不要停掉整個即時翻譯
                    log.warning("即時翻譯失敗：%s", e)
                    self.on_event({"type": "error", "message": f"翻譯失敗：{e}"})
            if self.log is not None:
                self.log.add({k: event.get(k) for k in ("id", "src", "start", "end", "text", "lang")}
                             | {"translation": translation})
