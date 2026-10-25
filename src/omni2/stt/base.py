"""STT 引擎介面。每個引擎只要實作 load() 和 transcribe()。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from omni2.config.settings import SttSettings
from omni2.core.types import Transcript
from omni2.vad.presets import VadPreset

ProgressFn = Callable[[float], None]


@dataclass
class TranscribeOptions:
    language: str | None = None  # None＝自動偵測
    vad: VadPreset | None = None
    hotwords: list[str] = field(default_factory=list)  # 術語表裡的原文，當作辨識提示
    beam_size: int = 5


class SttEngine(ABC):
    name: str = ""
    # True：transcribe_many 會把多個檔案放進同一次批次推論，呼叫端應該一次給多個檔案
    batches_files: bool = False

    def __init__(self, settings: SttSettings, live: bool = False):
        self.settings = settings
        # 即時模式只要文字、不要詞級時間戳，引擎可以少載一些東西、走比較快的路
        self.live = live

    def transcribe_text(self, audio: np.ndarray, language: str | None = None) -> tuple[str, str | None]:
        """即時模式用：一小段音訊（一句話）→（文字, 語言）。預設借用 transcribe。"""
        tr = self.transcribe(audio, TranscribeOptions(language=language,
                                                      vad=VadPreset(mode="fixed_window", window_ms=30000)))
        return " ".join(s.text for s in tr.segments).strip(), tr.language

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def transcribe(
        self, audio: np.ndarray, opts: TranscribeOptions, progress: ProgressFn | None = None
    ) -> Transcript: ...

    def transcribe_many(
        self,
        audios: list[np.ndarray],
        opts: TranscribeOptions,
        progress: Callable[[int, float], None] | None = None,
    ) -> list[Transcript]:
        """多個檔案一起辨識。能真正批次推論的引擎（Qwen3-ASR）會覆寫這個方法。"""
        results = []
        for i, audio in enumerate(audios):
            cb = (lambda p, i=i: progress(i, p)) if progress else None
            results.append(self.transcribe(audio, opts, cb))
        return results

    def close(self) -> None:
        pass
