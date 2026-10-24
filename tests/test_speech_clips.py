import numpy as np
import pytest

from omni2.stt import clips as fw
from omni2.vad.presets import PRESETS, VadPreset

SR = 16000


@pytest.fixture
def regions(monkeypatch):
    """用假的 VAD 結果取代 Silero，只測合併規則。"""
    holder = {}

    def fake(audio, options):
        return [{"start": int(s * SR), "end": int(e * SR)} for s, e in holder["value"]]

    import faster_whisper.vad

    monkeypatch.setattr(faster_whisper.vad, "get_speech_timestamps", fake)
    return holder


def test_long_utterances_stay_separate(regions):
    regions["value"] = [(0.0, 4.0), (5.0, 9.0), (9.8, 14.0)]
    clips = fw.speech_clips(np.zeros(SR), PRESETS["file"])
    assert [(c["start"], c["end"]) for c in clips] == [(0.0, 4.0), (5.0, 9.0), (9.8, 14.0)]


def test_short_clip_joins_next_when_gap_is_small(regions):
    regions["value"] = [(0.0, 1.2), (2.0, 5.0), (8.0, 8.5), (8.9, 12.0)]
    clips = fw.speech_clips(np.zeros(SR), PRESETS["file"])
    assert [(c["start"], c["end"]) for c in clips] == [(0.0, 5.0), (8.0, 12.0)]


def test_short_clip_does_not_join_across_long_pause(regions):
    regions["value"] = [(0.0, 1.0), (3.0, 6.0)]
    clips = fw.speech_clips(np.zeros(SR), PRESETS["file"])
    assert len(clips) == 2


def test_fixed_clips_cover_whole_audio():
    clips = fw.fixed_clips(20.0, VadPreset(mode="fixed_window", window_ms=8000))
    assert [(c["start"], c["end"]) for c in clips] == [(0.0, 8.0), (8.0, 16.0), (16.0, 20.0)]
