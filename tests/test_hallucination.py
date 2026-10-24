from omni2.core.types import Segment
from omni2.stt.hallucination import filter_segments, hallucination_reason


def test_blacklisted_phrases():
    assert hallucination_reason("Thank you for watching!")
    assert hallucination_reason("ご視聴ありがとうございました")
    assert hallucination_reason("字幕由 Amara.org 社群提供")


def test_short_text_with_high_no_speech():
    assert hallucination_reason("はい", no_speech_prob=0.9)
    assert hallucination_reason("はい", no_speech_prob=0.1) is None


def test_repetition_needs_corroboration():
    chant = "がんばれ " * 10
    assert hallucination_reason(chant, ratio=3.5, no_speech_prob=0.0) is None
    assert hallucination_reason(chant, ratio=3.5, no_speech_prob=0.5)
    assert hallucination_reason(chant, ratio=8.0, no_speech_prob=0.0)


def test_real_speech_is_kept():
    assert hallucination_reason("今日はいい天気ですね、散歩に行きましょう", no_speech_prob=0.02) is None


def test_filter_segments():
    segs = [Segment(0, 1, "Thank you."), Segment(1, 2, "本当にありがとう、助かりました"), Segment(2, 3, " ")]
    kept, dropped = filter_segments(segs)
    assert [s.text for s in kept] == ["本当にありがとう、助かりました"]
    assert len(dropped) == 2
