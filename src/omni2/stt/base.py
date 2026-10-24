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

    def __init__(self, settings: SttSettings):
        self.settings = settings

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
