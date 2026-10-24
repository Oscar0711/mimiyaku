"""OpenAI 語音辨識（gpt-4o-transcribe、whisper-1）。

逐段送出（沿用本地引擎的 VAD 切段），最多 4 段同時進行。whisper-1 有逐詞時間；
gpt-4o-transcribe 只回傳文字，逐詞時間依字寬比例估計。
※ 尚未用真實金鑰實測，只用模擬的回應測過。
"""

from __future__ import annotations

import numpy as np

from omni2.audio.decode import SAMPLE_RATE
from omni2.audio.encode import wav_bytes
from omni2.core import lang
from omni2.core.secrets import require_secret
from omni2.core.types import Segment, Transcript
from omni2.stt.base import ProgressFn, SttEngine, TranscribeOptions
from omni2.stt.cloud_clips import transcribe_clips
from omni2.stt.words import attach_punctuation, spread_words

# whisper-1 回傳的是語言全名
_LANGUAGE_NAMES = {"japanese": "ja", "english": "en", "korean": "ko", "chinese": "zh", "cantonese": "yue"}


class OpenAiEngine(SttEngine):
    name = "openai"

    def load(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=require_secret("openai", "OpenAI"), max_retries=3, timeout=120)
        self.model = self.settings.cloud.openai_model

    def transcribe(self, audio: np.ndarray, opts: TranscribeOptions,
                   progress: ProgressFn | None = None) -> Transcript:
        detected: list[str] = []

        def one(clip: np.ndarray, start: float, end: float) -> Segment | None:
            kwargs = {"model": self.model, "file": ("clip.wav", wav_bytes(clip), "audio/wav")}
            if opts.language:
                kwargs["language"] = lang.base(opts.language)
            if opts.hotwords:
                kwargs["prompt"] = "、".join(opts.hotwords)
            if self.model == "whisper-1":
                r = self.client.audio.transcriptions.create(
                    **kwargs, response_format="verbose_json", timestamp_granularities=["word"])
                text = (r.text or "").strip()
                items = [(w.word, start + w.start, start + w.end) for w in (r.words or [])]
                words = attach_punctuation(text, items) if items else spread_words(text, start, end)
                if getattr(r, "language", None):
                    detected.append(_LANGUAGE_NAMES.get(r.language.lower(), r.language))
            else:
                r = self.client.audio.transcriptions.create(**kwargs, response_format="json")
                text = (r.text or "").strip()
                words = spread_words(text, start, end)
            if not text:
                return None
            return Segment(start=start, end=end, text=text, words=words)

        segments = transcribe_clips(audio, opts, one, progress)
        if opts.language:
            language = opts.language
        elif detected:
            language = max(set(detected), key=detected.count)
        else:
            language = lang.guess_from_text("".join(s.text for s in segments))
        return Transcript(segments=segments, language=language, duration=len(audio) / SAMPLE_RATE)
