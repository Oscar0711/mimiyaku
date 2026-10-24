from omni2.config.settings import LayoutSettings
from omni2.core.types import Cue, Segment, Word
from omni2.post.layout import build_cues, fix_timing
from omni2.post.resegment import split_long
from omni2.post.text import atoms, join_texts, width, wrap


def test_width_counts_fullwidth_as_two():
    assert width("abc") == 3
    assert width("你好") == 4
    assert width("ＡＢ") == 4


def test_atoms_keep_punctuation_attached():
    assert atoms("「你好」，世界。") == ["「你", "好」，", "世", "界。"]
    assert atoms("Hello, world!") == ["Hello, ", "world!"]


def test_wrap_short_text_is_single_line():
    assert wrap("今天天氣很好", 42) == ["今天天氣很好"]


def test_wrap_cjk_is_balanced():
    text = "這是一句很長的字幕需要被切成兩行才能放得下而且兩行長度要接近"
    lines = wrap(text, 42)
    assert len(lines) == 2
    assert "".join(lines) == text
    assert abs(width(lines[0]) - width(lines[1])) <= 4


def test_wrap_latin_breaks_at_spaces():
    text = "This is a fairly long English subtitle line that must be wrapped somewhere"
    lines = wrap(text, 42)
    assert len(lines) == 2
    assert all(width(line) <= 42 for line in lines)
    assert " ".join(lines) == text


def test_wrap_never_drops_text():
    text = "字" * 120
    lines = wrap(text, 42)
    assert "".join(lines) == text


def test_join_texts():
    assert join_texts(["你好。", "再見。"]) == "你好。再見。"
    assert join_texts(["Hello.", "Bye."]) == "Hello. Bye."


def _seg(words):
    ws = [Word(s, e, t) for s, e, t in words]
    return Segment(ws[0].start, ws[-1].end, "".join(w.text for w in ws).strip(), ws)


def test_split_long_prefers_sentence_end():
    seg = _seg([
        (0.0, 1.0, " This"), (1.0, 2.0, " is"), (2.0, 3.0, " one."),
        (3.0, 4.0, " And"), (4.0, 5.0, " this"), (5.0, 6.0, " is"),
        (6.0, 7.0, " another"), (7.0, 8.0, " one"), (8.0, 9.0, " here."),
    ])
    parts = split_long([seg], max_duration=7.0, max_width=200)
    assert [p.text for p in parts] == ["This is one.", "And this is another one here."]
    assert parts[0].start == 0.0 and parts[-1].end == 9.0


def test_split_long_uses_whisper_space_marker_for_cjk():
    # 沒有標點、時間戳又把下一句的第一個字拉到停頓前面時，靠 Whisper 插入的空白判斷
    seg = _seg([
        (0.0, 1.0, "ありがとう"), (1.0, 2.0, "ございます"), (2.0, 2.2, " 今日"),
        (2.2, 3.0, "は"), (3.0, 4.0, "新しい"), (4.0, 5.0, "ゲーム"),
        (5.0, 6.0, "を"), (6.0, 7.0, "やります"), (7.0, 7.6, "実"), (9.0, 9.5, "は"),
    ])
    parts = split_long([seg], max_duration=7.0, max_width=200)
    assert parts[0].text == "ありがとうございます"


def test_split_long_keeps_short_segment():
    seg = _seg([(0.0, 1.0, "短"), (1.0, 2.0, "句")])
    assert split_long([seg], 7.0, 84) == [seg]


def test_fix_timing_extends_and_avoids_overlap():
    layout = LayoutSettings(min_duration=1.0, gap=0.1)
    cues = [Cue(0.0, 0.3, "嗨"), Cue(0.8, 3.0, "下一句"), Cue(2.5, 4.0, "重疊")]
    fix_timing(cues, layout)
    assert cues[0].end == 0.7  # 想延長到 1.0，但被下一句的開頭擋住
    assert cues[1].end == 2.4
    assert cues[2].end == 4.0


def test_build_cues_bilingual_falls_back_to_source():
    layout = LayoutSettings()
    segs = [Segment(0, 2, "こんにちは"), Segment(3, 5, "さようなら")]
    cues = build_cues(segs, ["你好", None], layout, bilingual=True)
    assert (cues[0].primary, cues[0].secondary) == ("你好", "こんにちは")
    assert (cues[1].primary, cues[1].secondary) == ("さようなら", None)
