from omni2.config.settings import LayoutSettings
from omni2.core.types import Cue, Segment
from omni2.export import formats as fmt


def test_times():
    assert fmt.srt_time(3723.456) == "01:02:03,456"
    assert fmt.vtt_time(0.5) == "00:00:00.500"
    assert fmt.ass_time(61.239) == "0:01:01.24"


def test_srt_bilingual():
    cues = [Cue(1.0, 2.5, "你好", "こんにちは"), Cue(3.0, 4.0, "再見")]
    out = fmt.to_srt(cues, LayoutSettings())
    assert out == (
        "1\n00:00:01,000 --> 00:00:02,500\n你好\nこんにちは\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n再見\n"
    )


def test_vtt_header_and_arrow_escape():
    out = fmt.to_vtt([Cue(0, 1, "a --> b")], LayoutSettings())
    assert out.startswith("WEBVTT\n")
    assert "a -> b" in out


def test_ass_secondary_style_and_braces():
    out = fmt.to_ass([Cue(0, 1, "譯{x}", "原文")], LayoutSettings())
    assert r"譯｛x｝\N{\rSecondary}原文" in out


def test_txt_with_speakers_and_translation():
    segs = [Segment(1, 2, "hello", speaker="SPEAKER_01"), Segment(3, 4, "hi", speaker="SPEAKER_00")]
    out = fmt.to_txt(segs, ["哈囉", "嗨"])
    assert out.splitlines() == [
        "[00:00:01] 講者 A：hello",
        "           哈囉",
        "[00:00:03] 講者 B：hi",
        "           嗨",
    ]


def test_md_groups_speaker_turns():
    segs = [
        Segment(0, 1, "早安。", speaker="S1"),
        Segment(1, 2, "今天好熱。", speaker="S1"),
        Segment(2, 3, "對啊。", speaker="S2"),
    ]
    out = fmt.to_md(segs, None, title="會議")
    assert "**講者 A** · 00:00:00\n\n早安。今天好熱。" in out
    assert "**講者 B** · 00:00:02\n\n對啊。" in out
