"""即時切句：把連續的音訊切成「一句一句」，並在講話期間定期送出目前這句（給兩段式字幕的原文草稿）。

規則：
* 人聲機率超過 threshold 開始一句，往前補 pad_ms 的音訊（避免切掉開頭的子音）。
* 句中機率低於 end_threshold 連續 min_silence_ms，這句結束。兩個門檻是遲滯設計，
  句中的輕聲、換氣不會被當成句子結束。
* 講話期間每 partial_ms 送出一次目前累積的音訊（草稿），讓原文能邊聽邊出現。
* 一句超過 max_ms 還沒結束，就在最近 2 秒內機率最低（最像換氣）的地方切開，剩下的接成下一句。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from omni2.vad.presets import VadPreset
from omni2.vad.stream import FRAME, StreamingVad

SR = 16000
_FRAME_MS = FRAME * 1000 / SR


@dataclass
class Utterance:
    id: int
    start: float  # 從開始擷取算起的秒數
    audio: np.ndarray
    final: bool


@dataclass
class _Current:
    start_frame: int
    frames: list[np.ndarray] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)


class Segmenter:
    def __init__(self, preset: VadPreset, partial_ms: int = 600, max_ms: int = 8000, vad=None):
        self.p = preset
        self.partial_frames = max(1, int(partial_ms / _FRAME_MS))
        self.max_frames = int(max_ms / _FRAME_MS)
        self.pad_frames = max(1, int(preset.speech_pad_ms / _FRAME_MS))
        self.silence_frames = max(1, int(preset.min_silence_ms / _FRAME_MS))
        self.min_speech_frames = max(1, int(preset.min_speech_ms / _FRAME_MS))
        self.vad = vad or StreamingVad()
        self.frame_no = 0
        # 還沒開始講話時的最近幾格（含這一格），開頭往前補用；要記住每格真正的機率，
        # 否則補上的幾格會被當成語音，連很短的雜音都會變成一句話
        self.history: list[tuple[np.ndarray, float]] = []
        self.cur: _Current | None = None
        self.silence = 0
        self.since_partial = 0
        self.next_id = 0

    def feed(self, audio: np.ndarray) -> list[Utterance]:
        out: list[Utterance] = []
        for frame, prob in self.vad(audio):
            self.frame_no += 1
            if self.cur is None:
                self.history = (self.history + [(frame, prob)])[-(self.pad_frames + 1):]
                if prob >= self.p.threshold:
                    self.cur = _Current(self.frame_no - len(self.history), [f for f, _ in self.history],
                                        [p for _, p in self.history])
                    self.history = []
                    self.silence = 0
                    self.since_partial = 0
                continue
            self.cur.frames.append(frame)
            self.cur.probs.append(prob)
            self.silence = self.silence + 1 if prob < self.p.end_threshold else 0
            self.since_partial += 1
            if self.silence >= self.silence_frames:
                out += self._finish(len(self.cur.frames) - self.silence + self.pad_frames)
            elif len(self.cur.frames) >= self.max_frames:
                out += self._finish(self._breath_point(), carry=True)
            elif self.since_partial >= self.partial_frames:
                self.since_partial = 0
                out.append(self._utterance(len(self.cur.frames), final=False))
        return out

    def flush(self) -> list[Utterance]:
        """停止擷取時，把講到一半的句子當成完整的一句送出。"""
        return self._finish(len(self.cur.frames)) if self.cur else []

    def _breath_point(self) -> int:
        """最近 2 秒內機率最低的那一格（優先選比較後面的）。"""
        probs = self.cur.probs
        lo = max(self.min_speech_frames, len(probs) - int(2000 / _FRAME_MS))
        window = probs[lo:]
        k = len(window) - 1 - int(np.argmin(window[::-1]))
        return lo + k + 1

    def _utterance(self, n_frames: int, final: bool) -> Utterance:
        audio = np.concatenate(self.cur.frames[:n_frames]) if n_frames else np.zeros(0, np.float32)
        return Utterance(self.next_id, self.cur.start_frame * FRAME / SR, audio, final)

    def _finish(self, n_frames: int, carry: bool = False) -> list[Utterance]:
        cur = self.cur
        n_frames = min(n_frames, len(cur.frames))
        out = []
        speech = sum(p >= self.p.end_threshold for p in cur.probs[:n_frames])
        if speech >= self.min_speech_frames:
            out.append(self._utterance(n_frames, final=True))
            self.next_id += 1
        if carry and n_frames < len(cur.frames):  # 被硬切的長句：剩下的接成下一句
            self.cur = _Current(cur.start_frame + n_frames, cur.frames[n_frames:], cur.probs[n_frames:])
            self.since_partial = 0
        else:
            self.cur = None
            self.history = []
        self.silence = 0
        return out
