"""把片段和譯文排成字幕：決定每則字幕顯示的文字和時間。"""

from __future__ import annotations

from omni2.config.settings import LayoutSettings
from omni2.core.types import Cue, Segment
from omni2.post.text import width, wrap


def build_cues(
    segments: list[Segment],
    translations: list[str | None] | None,
    layout: LayoutSettings,
    *,
    bilingual: bool = False,
) -> list[Cue]:
    cues: list[Cue] = []
    for i, seg in enumerate(segments):
        translated = translations[i] if translations else None
        primary = translated if translated else seg.text
        secondary = seg.text if bilingual and translated else None
        cues.append(Cue(seg.start, seg.end, primary, secondary, seg.speaker))
    fix_timing(cues, layout)
    return cues


def fix_timing(cues: list[Cue], layout: LayoutSettings) -> None:
    """拉長太短、讀不完的字幕，再把和下一則重疊的部分收回來。"""
    cues.sort(key=lambda c: c.start)
    for cue in cues:
        reading = width(cue.primary) / 2 / layout.max_cps
        needed = max(layout.min_duration, reading)
        if cue.end - cue.start < needed:
            cue.end = cue.start + needed
    for cur, nxt in zip(cues, cues[1:]):
        limit = nxt.start - layout.gap
        if cur.end > limit:
            # 下一則來得太快時，至少保留一小段顯示時間
            cur.end = max(limit, cur.start + 0.2)
    for cue in cues:
        cue.start, cue.end = round(cue.start, 3), round(cue.end, 3)


def cue_lines(cue: Cue, layout: LayoutSettings) -> tuple[list[str], list[str]]:
    primary = wrap(cue.primary, layout.max_line_width)
    secondary = wrap(cue.secondary, layout.max_line_width) if cue.secondary else []
    return primary, secondary
