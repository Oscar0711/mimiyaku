"""即時辨識器：吃連續的音訊，吐出「草稿」（講到一半）和「確定」（一句結束）兩種結果。

* LocalRecognizer：切句器＋本地引擎。草稿是把目前這句重新辨識一次；只有在來得及的時候
  才做（落後時跳過草稿，確定的句子一定會做）。
* FunAsrRealtimeRecognizer：阿里雲 Fun-ASR 串流辨識，原生就會回傳逐步修正的結果。
每個事件：{"type": "partial"|"final", "id", "start", "end", "text", "lang"}，時間是從開始擷取
算起的秒數。
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from omni2.live.segmenter import Segmenter
from omni2.stt.base import SttEngine
from omni2.vad.presets import VadPreset

log = logging.getLogger(__name__)
SR = 16000


class LocalRecognizer:
    def __init__(self, engine: SttEngine, preset: VadPreset, label: str, language: str | None,
                 partial_ms: int, max_ms: int):
        self.engine = engine
        self.segmenter = Segmenter(preset, partial_ms=partial_ms, max_ms=max_ms)
        self.label = label
        self.language = language

    def feed(self, audio: np.ndarray, behind: bool = False) -> list[dict]:
        """behind：佇列裡還有很多沒處理的音訊，這次就不做草稿。"""
        events = []
        for u in self.segmenter.feed(audio):
            if not u.final and behind:
                continue
            events.append(self._decode(u))
        return [e for e in events if e]

    def flush(self) -> list[dict]:
        return [e for e in (self._decode(u) for u in self.segmenter.flush()) if e]

    def _decode(self, u) -> dict | None:
        t = time.perf_counter()
        text, language = self.engine.transcribe_text(u.audio, self.language)
        if not text and not u.final:
            return None
        return {"type": "final" if u.final else "partial", "id": f"{self.label}-{u.id}", "src": self.label,
                "start": round(u.start, 2), "end": round(u.start + len(u.audio) / SR, 2), "text": text,
                "lang": language, "decode_s": round(time.perf_counter() - t, 3)}


class FunAsrRealtimeRecognizer:
    def __init__(self, label: str, model: str, region: str, language: str | None):
        from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

        from omni2.core.secrets import require_secret
        from omni2.stt.funasr_engine import configure_region

        configure_region(region)
        self.label = label
        self.language = language
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._n = 0
        outer = self

        class Callback(RecognitionCallback):
            def on_event(self, result: RecognitionResult) -> None:
                sentence = result.get_sentence()
                if not sentence or "text" not in sentence:
                    return
                final = RecognitionResult.is_sentence_end(sentence)
                outer._push(sentence, final)

            def on_error(self, result: RecognitionResult) -> None:
                with outer._lock:
                    outer._events.append({"type": "error", "src": label,
                                          "message": f"Fun-ASR 串流錯誤：{result.message}"})

        import dashscope

        # 串流辨識沒有 api_key 參數，只讀全域設定；這段只在辨識子行程裡執行
        dashscope.api_key = require_secret("dashscope", "阿里雲 DashScope")
        kwargs = {"language_hints": [language.split("-")[0]]} if language else {}
        self.recognition = Recognition(model=model, callback=Callback(), format="pcm", sample_rate=SR, **kwargs)
        self.recognition.start()

    def _push(self, sentence: dict, final: bool) -> None:
        with self._lock:
            event = {"type": "final" if final else "partial", "id": f"{self.label}-{self._n}", "src": self.label,
                     "start": round((sentence.get("begin_time") or 0) / 1000, 2),
                     "end": round((sentence.get("end_time") or sentence.get("begin_time") or 0) / 1000, 2),
                     "text": self._tidy((sentence.get("text") or "").strip()), "lang": self.language}
            if final:
                self._n += 1
            self._events.append(event)

    def _tidy(self, text: str) -> str:
        """和檔案模式一樣修掉 Fun-ASR 的小毛病（日文多餘的空格、英文大小寫）。"""
        from omni2.core import lang
        from omni2.stt.funasr_engine import tidy_text

        return tidy_text(text, self.language or lang.guess_from_text(text))

    def feed(self, audio: np.ndarray, behind: bool = False) -> list[dict]:
        self.recognition.send_audio_frame((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        with self._lock:
            events, self._events = self._events, []
        return events

    def flush(self) -> list[dict]:
        try:
            self.recognition.stop()
        except Exception as e:  # noqa: BLE001 — 已經停了
            log.info("Fun-ASR 串流停止：%s", e)
        with self._lock:
            events, self._events = self._events, []
        return events
