"""VAD 情境預設。數值沿用上一版對真實音訊調過的結果。

threshold／end_threshold 是一組遲滯門檻：前者是「開始一句話」要達到的機率，後者是
「維持在這句話裡」的機率。無聲子音、中文輕聲、日文無聲化母音都會讓機率在句中掉到
開始門檻以下，只用單一門檻會被誤判成句子結束。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VadPreset:
    mode: str = "adaptive"  # adaptive＝依 VAD 切段；fixed_window＝固定窗口（音樂）
    threshold: float = 0.5
    end_threshold: float = 0.35
    min_silence_ms: int = 500
    min_speech_ms: int = 250
    speech_pad_ms: int = 100
    max_segment_ms: int = 15000
    window_ms: int = 8000
    overlap_ms: int = 1000


PRESETS: dict[str, VadPreset] = {
    # 一般對話、會議
    "conversation": VadPreset(threshold=0.5, end_threshold=0.35, min_silence_ms=500,
                              min_speech_ms=250, speech_pad_ms=100, max_segment_ms=15000),
    # 直播、遊戲：重視低延遲，短停頓就切
    "live": VadPreset(threshold=0.45, end_threshold=0.3, min_silence_ms=250,
                      min_speech_ms=200, speech_pad_ms=80, max_segment_ms=8000),
    # ASMR、氣音：輕聲貼著門檻，兩個門檻要拉開，否則每次換氣都會斷句
    "asmr": VadPreset(threshold=0.25, end_threshold=0.15, min_silence_ms=700,
                      min_speech_ms=300, speech_pad_ms=200, max_segment_ms=20000),
    # 音樂、歌唱：VAD 不可靠，改用重疊的固定窗口
    "music": VadPreset(mode="fixed_window", window_ms=8000, overlap_ms=1000),
    # 預錄檔案：偏向準確度的切法
    "file": VadPreset(threshold=0.4, end_threshold=0.28, min_silence_ms=600,
                      min_speech_ms=300, speech_pad_ms=150, max_segment_ms=30000),
}


def get_preset(name: str) -> VadPreset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"未知的 VAD 預設：{name}（可用：{', '.join(PRESETS)}）") from None
