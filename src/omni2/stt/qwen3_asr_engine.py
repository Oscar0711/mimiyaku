"""Qwen3-ASR（transformers）加 Qwen3-ForcedAligner 的詞級時間戳。

和 faster-whisper 一樣先用 VAD 切成語音區段；不同的是，多個檔案的所有區段會一起送進
同一次批次推論，檔案越多越划算。Qwen3-ASR 會自己加標點，日文、中文的斷句比 Whisper 好。
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable

import numpy as np

from omni2.audio.decode import SAMPLE_RATE
from omni2.core.types import Segment, Transcript
from omni2.stt.base import ProgressFn, SttEngine, TranscribeOptions
from omni2.stt.clips import make_clips
from omni2.stt.words import attach_punctuation

log = logging.getLogger(__name__)

# 每次送進模型的區段數；同時決定進度回報的頻率
_CHUNK = 32

_NAMES = {"ja": "Japanese", "en": "English", "ko": "Korean", "zh": "Chinese", "yue": "Cantonese",
          "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian", "th": "Thai",
          "vi": "Vietnamese", "id": "Indonesian", "it": "Italian", "pt": "Portuguese"}
_CODES = {v: k for k, v in _NAMES.items()}


class Qwen3AsrEngine(SttEngine):
    name = "qwen3-asr"
    batches_files = True

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        s = self.settings
        # 即時模式不需要詞級時間戳，不載入對齊器（省約 1.2 GB 顯示記憶體）
        aligner = None if self.live else s.qwen_aligner
        log.info("載入 Qwen3-ASR %s%s", s.qwen_model, f"＋{aligner}" if aligner else "")
        device = "cuda:0" if s.device == "cuda" else s.device
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self._model = Qwen3ASRModel.from_pretrained(
            s.qwen_model,
            forced_aligner=aligner,
            forced_aligner_kwargs={"dtype": dtype, "device_map": device} if aligner else None,
            max_inference_batch_size=_CHUNK,
            dtype=dtype,
            device_map=device,
        )

    def transcribe_text(self, audio: np.ndarray, language: str | None = None) -> tuple[str, str | None]:
        name = _NAMES.get((language or "").split("-")[0]) if language else None
        result = self._model.transcribe(audio=[(audio, SAMPLE_RATE)], language=name, return_time_stamps=False)[0]
        detected = (result.language or "").split(",")[0].strip()
        return (result.text or "").strip(), language or _CODES.get(detected)

    def transcribe(self, audio: np.ndarray, opts: TranscribeOptions,
                   progress: ProgressFn | None = None) -> Transcript:
        cb = (lambda _i, p: progress(p)) if progress else None
        return self.transcribe_many([audio], opts, cb)[0]

    def transcribe_many(self, audios: list[np.ndarray], opts: TranscribeOptions,
                        progress: Callable[[int, float], None] | None = None) -> list[Transcript]:
        jobs = []  # (檔案編號, 區段)
        for i, audio in enumerate(audios):
            for clip in make_clips(audio, opts.vad):
                jobs.append((i, clip))
        per_file_total = Counter(i for i, _ in jobs)
        per_file_done: Counter = Counter()
        language = _NAMES.get((opts.language or "").split("-")[0]) if opts.language else None
        context = "、".join(opts.hotwords)

        segments: list[list[Segment]] = [[] for _ in audios]
        langs: list[Counter] = [Counter() for _ in audios]
        for pos in range(0, len(jobs), _CHUNK):
            batch = jobs[pos : pos + _CHUNK]
            wavs = [(audios[i][int(c["start"] * SAMPLE_RATE): int(c["end"] * SAMPLE_RATE)], SAMPLE_RATE)
                    for i, c in batch]
            results = self._model.transcribe(audio=wavs, context=context, language=language,
                                             return_time_stamps=True)
            for (i, clip), r in zip(batch, results):
                per_file_done[i] += 1
                text = (r.text or "").strip()
                if text:
                    items = [(it.text, clip["start"] + it.start_time, clip["start"] + it.end_time)
                             for it in (r.time_stamps or [])]
                    words = attach_punctuation(text, items)
                    start = words[0].start if words else clip["start"]
                    end = words[-1].end if words else clip["end"]
                    segments[i].append(Segment(start=start, end=end, text=text, words=words))
                    for name in (r.language or "").split(","):
                        if name.strip():
                            langs[i][name.strip()] += 1
                if progress:
                    progress(i, per_file_done[i] / per_file_total[i])

        out = []
        for i, audio in enumerate(audios):
            detected = langs[i].most_common(1)[0][0] if langs[i] else None
            code = opts.language or _CODES.get(detected or "", None)
            out.append(Transcript(segments=segments[i], language=code, duration=len(audio) / SAMPLE_RATE))
        return out

    def close(self) -> None:
        self._model = None
