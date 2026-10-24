"""雲端引擎共用：用 VAD 切成語音區段，平行送出，依序組回來。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np

from omni2.audio.decode import SAMPLE_RATE
from omni2.core.types import Segment
from omni2.stt.base import ProgressFn, TranscribeOptions
from omni2.stt.clips import make_clips

# 送一段：（這段的音訊, 這段在整個檔案裡的起訖秒數）→ 片段（沒有內容時回傳 None）
ClipFn = Callable[[np.ndarray, float, float], "Segment | None"]


def transcribe_clips(audio: np.ndarray, opts: TranscribeOptions, fn: ClipFn,
                     progress: ProgressFn | None = None, workers: int = 4) -> list[Segment]:
    clips = make_clips(audio, opts.vad)
    results: list[Segment | None] = [None] * len(clips)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fn, audio[int(c["start"] * SAMPLE_RATE): int(c["end"] * SAMPLE_RATE)],
                        c["start"], c["end"]): i
            for i, c in enumerate(clips)
        }
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if progress:
                progress(done / len(clips))
    return [s for s in results if s is not None and s.text.strip()]
