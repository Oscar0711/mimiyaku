"""把整段音訊切成要送進 STT 的語音區段。faster-whisper 和 Qwen3-ASR 共用。

每個區段各自解碼並保留區段內的停頓。faster-whisper 內建的 VAD 合併會把多段語音直接
接起來、拿掉中間的靜音，模型聽到的句子黏在一起：日文因此完全沒有標點，詞級時間戳在
句子交界處也會錯位（見 docs/design.md 的實作紀錄）。
"""

from __future__ import annotations

import numpy as np

from omni2.audio.decode import SAMPLE_RATE
from omni2.vad.presets import VadPreset

# Whisper 一次最多看 30 秒
MAX_CLIP_S = 28.0
# 太短的區段單獨辨識時上下文不足，併入前一段（只要中間的停頓不長）
MIN_CLIP_S = 3.0
MAX_JOIN_GAP_S = 1.5


def speech_clips(audio: np.ndarray, vad: VadPreset) -> list[dict]:
    """回傳 [{"start": 秒, "end": 秒}]。"""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    max_clip = min(vad.max_segment_ms / 1000, MAX_CLIP_S)
    options = VadOptions(
        threshold=vad.threshold,
        neg_threshold=vad.end_threshold,
        min_speech_duration_ms=vad.min_speech_ms,
        max_speech_duration_s=max_clip,
        min_silence_duration_ms=vad.min_silence_ms,
        speech_pad_ms=vad.speech_pad_ms,
    )
    clips: list[dict] = []
    for region in get_speech_timestamps(audio, options):
        start, end = region["start"] / SAMPLE_RATE, region["end"] / SAMPLE_RATE
        if clips:
            last = clips[-1]
            if (last["end"] - last["start"] < MIN_CLIP_S
                    and start - last["end"] <= MAX_JOIN_GAP_S
                    and end - last["start"] <= max_clip):
                last["end"] = end
                continue
        clips.append({"start": start, "end": end})
    return clips


def fixed_clips(duration: float, vad: VadPreset) -> list[dict]:
    """音樂模式：VAD 不可靠，改用固定長度的窗口。"""
    step = vad.window_ms / 1000
    clips, t = [], 0.0
    while t < duration:
        clips.append({"start": t, "end": min(duration, t + step)})
        t += step
    return clips


def make_clips(audio: np.ndarray, vad: VadPreset | None) -> list[dict]:
    vad = vad or VadPreset()
    if vad.mode == "fixed_window":
        return fixed_clips(len(audio) / SAMPLE_RATE, vad)
    return speech_clips(audio, vad)
