"""faster-whisper（CTranslate2）。區段切法見 omni2.stt.clips，每個區段透過 clip_timestamps
各自送進批次推論。"""

from __future__ import annotations

import inspect
import logging

import numpy as np

from omni2.audio.decode import SAMPLE_RATE
from omni2.core import lang
from omni2.core.types import Segment, Transcript, Word
from omni2.stt.base import ProgressFn, SttEngine, TranscribeOptions
from omni2.stt.clips import make_clips

log = logging.getLogger(__name__)


def _accepted(fn, kwargs: dict) -> dict:
    """只傳函式認得的參數，faster-whisper 小版本之間參數會增減。"""
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


class FasterWhisperEngine(SttEngine):
    name = "faster-whisper"

    def load(self) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        s = self.settings
        log.info("載入 faster-whisper %s（%s, %s）", s.model, s.device, s.compute_type)
        self._model = WhisperModel(s.model, device=s.device, compute_type=s.compute_type)
        self._pipeline = BatchedInferencePipeline(self._model)

    def transcribe_text(self, audio: np.ndarray, language: str | None = None) -> tuple[str, str | None]:
        # 即時模式用 greedy：每句都很短，beam search 的準確度好處不大，卻讓延遲多出約 0.3 秒
        segments, info = self._model.transcribe(
            audio, language=lang.stt_code(language), beam_size=1 if self.live else self.settings.beam_size,
            vad_filter=False, condition_on_previous_text=False, without_timestamps=True)
        return "".join(s.text for s in segments).strip(), info.language

    def transcribe(
        self, audio: np.ndarray, opts: TranscribeOptions, progress: ProgressFn | None = None
    ) -> Transcript:
        duration = len(audio) / SAMPLE_RATE
        clips = make_clips(audio, opts.vad)
        if not clips:
            return Transcript(segments=[], language=lang.stt_code(opts.language), duration=duration)
        log.info("%d 個語音區段，共 %.1f 秒", len(clips), sum(c["end"] - c["start"] for c in clips))

        fn = self._pipeline.transcribe
        kwargs = {
            "language": lang.stt_code(opts.language),
            "beam_size": opts.beam_size,
            "batch_size": max(1, self.settings.batch_size),
            "word_timestamps": True,
            # 關掉「以前文為條件」：前一段的幻覺不會一路傳下去
            "condition_on_previous_text": False,
            "hotwords": " ".join(opts.hotwords) or None,
            "clip_timestamps": clips,
        }
        seg_iter, info = fn(audio, **_accepted(fn, kwargs))

        segments: list[Segment] = []
        for s in seg_iter:
            words = [Word(w.start, w.end, w.word, w.probability) for w in (s.words or [])]
            segments.append(
                Segment(
                    start=s.start,
                    end=s.end,
                    text=s.text.strip(),
                    words=words,
                    no_speech_prob=s.no_speech_prob,
                    compression_ratio=s.compression_ratio,
                    avg_logprob=s.avg_logprob,
                )
            )
            if progress:
                progress(min(1.0, s.end / duration))
        return Transcript(segments=segments, language=info.language, duration=duration)

    def close(self) -> None:
        self._pipeline = None
        self._model = None
