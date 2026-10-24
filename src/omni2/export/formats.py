"""輸出格式：字幕（SRT、VTT、ASS）用排好的 Cue，逐字稿（TXT、MD）直接用片段。"""

from __future__ import annotations

from omni2.config.settings import LayoutSettings
from omni2.core.types import Cue, Segment
from omni2.post.layout import cue_lines
from omni2.post.text import join_texts

SUBTITLE_FORMATS = ("srt", "vtt", "ass")
TRANSCRIPT_FORMATS = ("txt", "md")
ALL_FORMATS = SUBTITLE_FORMATS + TRANSCRIPT_FORMATS


def _hms(t: float) -> tuple[int, int, int, int]:
    ms = max(0, round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return h, m, s, ms


def srt_time(t: float) -> str:
    h, m, s, ms = _hms(t)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def vtt_time(t: float) -> str:
    h, m, s, ms = _hms(t)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def ass_time(t: float) -> str:
    cs = max(0, round(t * 100))  # ASS 只到百分之一秒
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def clock(t: float) -> str:
    h, m, s, _ = _hms(t)
    return f"{h:02}:{m:02}:{s:02}"


def to_srt(cues: list[Cue], layout: LayoutSettings) -> str:
    blocks = []
    for i, cue in enumerate(cues, 1):
        primary, secondary = cue_lines(cue, layout)
        body = "\n".join(primary + secondary)
        blocks.append(f"{i}\n{srt_time(cue.start)} --> {srt_time(cue.end)}\n{body}\n")
    return "\n".join(blocks)


def to_vtt(cues: list[Cue], layout: LayoutSettings) -> str:
    blocks = ["WEBVTT\n"]
    for cue in cues:
        primary, secondary = cue_lines(cue, layout)
        body = "\n".join(primary + secondary).replace("-->", "->")
        blocks.append(f"{vtt_time(cue.start)} --> {vtt_time(cue.end)}\n{body}\n")
    return "\n".join(blocks)


_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft JhengHei,64,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,50,1
Style: Secondary,Microsoft JhengHei,44,&H00D0D0D0,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,1,2,60,60,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_escape(line: str) -> str:
    # 大括號在 ASS 裡是特效標籤，換成全形避免被吃掉
    return line.replace("\\", "＼").replace("{", "｛").replace("}", "｝")


def to_ass(cues: list[Cue], layout: LayoutSettings) -> str:
    events = []
    for cue in cues:
        primary, secondary = cue_lines(cue, layout)
        text = r"\N".join(_ass_escape(x) for x in primary)
        if secondary:
            text += r"\N{\rSecondary}" + r"\N".join(_ass_escape(x) for x in secondary)
        events.append(f"Dialogue: 0,{ass_time(cue.start)},{ass_time(cue.end)},Default,,0,0,0,,{text}")
    return _ASS_HEADER + "\n".join(events) + "\n"


def speaker_labels(segments: list[Segment]) -> dict[str, str]:
    """把 SPEAKER_00 這類代號依出場順序換成「講者 A、B…」。"""
    labels: dict[str, str] = {}
    for seg in segments:
        if seg.speaker and seg.speaker not in labels:
            idx = len(labels)
            letter = chr(ord("A") + idx) if idx < 26 else str(idx + 1)
            labels[seg.speaker] = f"講者 {letter}"
    return labels


def to_txt(
    segments: list[Segment],
    translations: list[str | None] | None,
    *,
    timestamps: bool = True,
) -> str:
    labels = speaker_labels(segments)
    lines = []
    for i, seg in enumerate(segments):
        head = f"[{clock(seg.start)}] " if timestamps else ""
        who = f"{labels[seg.speaker]}：" if seg.speaker in labels else ""
        lines.append(f"{head}{who}{seg.text}")
        translated = translations[i] if translations else None
        if translated:
            lines.append(f"{' ' * len(head)}{translated}")
    return "\n".join(lines) + "\n"


def to_md(
    segments: list[Segment],
    translations: list[str | None] | None,
    *,
    title: str,
    timestamps: bool = True,
) -> str:
    """有講者時，同一位講者連續說的話併成一段；沒有講者時一句一行。"""
    labels = speaker_labels(segments)
    out = [f"# {title}", ""]
    if labels:
        turns: list[tuple[Segment, list[int]]] = []
        for i, seg in enumerate(segments):
            if turns and turns[-1][0].speaker == seg.speaker:
                turns[-1][1].append(i)
            else:
                turns.append((seg, [i]))
        for first, idx in turns:
            head = f"**{labels.get(first.speaker, '講者 ?')}**"
            if timestamps:
                head += f" · {clock(first.start)}"
            out += [head, "", join_texts([segments[i].text for i in idx])]
            if translations:
                translated = join_texts([translations[i] or "" for i in idx])
                if translated:
                    out += ["", f"> {translated}"]
            out.append("")
    else:
        for i, seg in enumerate(segments):
            head = f"`{clock(seg.start)}` " if timestamps else ""
            out.append(f"{head}{seg.text}  ")
            translated = translations[i] if translations else None
            if translated:
                out.append(f"> {translated}")
                out.append("")
        out.append("")
    return "\n".join(out)
