"""Qwen3-ASR（transformers）加 Qwen3-ForcedAligner 的詞級時間戳。

和 faster-whisper 一樣先用 VAD 切成語音區段；不同的是，多個檔案的所有區段會一起送進
同一次批次推論，檔案越多越划算。Qwen3-ASR 會自己加標點，日文、中文的斷句比 Whisper 好。
"""

from __future__ import annotations

import logging
import unicodedata
from collections import Counter
from typing import Callable

import numpy as np

from omni2.audio.decode import SAMPLE_RATE
from omni2.core.types import Segment, Transcript, Word
from omni2.stt.base import ProgressFn, SttEngine, TranscribeOptions
from omni2.stt.clips import make_clips

log = logging.getLogger(__name__)

# 每次送進模型的區段數；同時決定進度回報的頻率
_CHUNK = 32

_NAMES = {"ja": "Japanese", "en": "English", "ko": "Korean", "zh": "Chinese", "yue": "Cantonese",
          "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian", "th": "Thai",
          "vi": "Vietnamese", "id": "Indonesian", "it": "Italian", "pt": "Portuguese"}
_CODES = {v: k for k, v in _NAMES.items()}
# 開括號屬於下一個詞，不要接在前一個詞後面
_OPENING = set("「『（〈《【〔“‘([{")


def _is_word_char(ch: str) -> bool:
    return ch == "'" or unicodedata.category(ch)[0] in "LN"


def attach_punctuation(text: str, items: list[tuple[str, float, float]]) -> list[Word]:
    """對齊器輸出的詞去掉了標點，這裡依序對回原文，把標點和空白接回每個詞。

    每個詞的文字包含它前面的空白與開括號、後面的標點，串起來就是原文。重新切段要靠
    「。」這類句尾標點，所以不能少。
    """
    words: list[Word] = []
    pos = 0
    folded = text.casefold()
    for token, start, end in items:
        idx = folded.find(token.casefold(), pos) if token else -1
        if idx < 0:
            words.append(Word(start, end, token))
            continue
        stop = idx + len(token)
        while (stop < len(text) and not text[stop].isspace() and not _is_word_char(text[stop])
               and text[stop] not in _OPENING):
            stop += 1
        words.append(Word(start, end, text[pos:stop]))
        pos = stop
    if words and pos < len(text):
        last = words[-1]
        words[-1] = Word(last.start, last.end, last.text + text[pos:])
    return words


class Qwen3AsrEngine(SttEngine):
    name = "qwen3-asr"
    batches_files = True

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        s = self.settings
        log.info("載入 Qwen3-ASR %s＋%s", s.qwen_model, s.qwen_aligner)
        device = "cuda:0" if s.device == "cuda" else s.device
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self._model = Qwen3ASRModel.from_pretrained(
            s.qwen_model,
            forced_aligner=s.qwen_aligner,
            forced_aligner_kwargs={"dtype": dtype, "device_map": device},
            max_inference_batch_size=_CHUNK,
            dtype=dtype,
            device_map=device,
        )

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
