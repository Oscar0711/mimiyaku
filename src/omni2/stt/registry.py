"""引擎名稱對應到實作。延遲 import，沒用到的引擎不會載入它的相依套件。"""

from __future__ import annotations

import importlib

from omni2.config.settings import SttSettings
from omni2.stt.base import SttEngine

ENGINES = {
    "faster-whisper": "omni2.stt.faster_whisper_engine:FasterWhisperEngine",
    "qwen3-asr": "omni2.stt.qwen3_asr_engine:Qwen3AsrEngine",
    "funasr": "omni2.stt.funasr_engine:FunAsrEngine",
    "openai": "omni2.stt.openai_engine:OpenAiEngine",
    "google": "omni2.stt.google_engine:GoogleEngine",
}


def create_engine(settings: SttSettings) -> SttEngine:
    try:
        target = ENGINES[settings.engine]
    except KeyError:
        raise ValueError(f"未知的 STT 引擎：{settings.engine}（可用：{', '.join(ENGINES)}）") from None
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)(settings)
