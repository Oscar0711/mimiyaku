"""在翻譯之前，用詞級時間戳把過長的片段切成字幕大小。

在翻譯前切，原文和譯文才能一對一對應；翻譯完才切，就得猜譯文該從哪裡斷。
"""

from __future__ import annotations

from omni2.core.types import Segment
from omni2.post.text import is_wide, width

_SENTENCE_END = ("。", "！", "？", "!", "?", "…", ".")
_CLAUSE = ("、", "，", ",", ";", "；", ":", "：")


def split_long(segments: list[Segment], max_duration: float, max_width: int) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        out.extend(_split(seg, max_duration, max_width))
    return out


def _split(seg: Segment, max_duration: float, max_width: int) -> list[Segment]:
    fits = seg.duration <= max_duration and width(seg.text) <= max_width
    if fits or len(seg.words) < 2:
        return [seg]
    i = _best_boundary(seg)
    left, right = seg.words[:i], seg.words[i:]
    cut = (left[-1].end + right[0].start) / 2
    return _split(_make(seg, left, seg.start, cut), max_duration, max_width) + _split(
        _make(seg, right, cut, seg.end), max_duration, max_width
    )


def _best_boundary(seg: Segment) -> int:
    """在第 i 個詞之前切。偏好句尾標點、其次子句標點和長停頓，同時不要切得太偏。"""
    words = seg.words
    total = sum(width(w.text.strip()) for w in words) or 1
    best_i, best_score = 1, float("-inf")
    acc = 0
    for i in range(1, len(words)):
        prev = words[i - 1].text.strip()
        acc += width(prev)
        score = 0.0
        if prev.endswith(_SENTENCE_END):
            score += 3.0
        elif prev.endswith(_CLAUSE):
            score += 1.5
        pause = max(0.0, words[i].start - words[i - 1].end)
        score += min(pause, 1.0) * 3.0
        # 日文、中文沒有標點時，Whisper 會在片語邊界插入空白，這是比時間戳可靠的訊號
        # （時間戳在句子交界處常把下一句的第一個字拉到停頓前面）
        if prev and is_wide(prev[-1]) and words[i].text.startswith(" "):
            score += 2.0
        score -= abs(acc / total - 0.5) * 4.0
        if score > best_score:
            best_i, best_score = i, score
    return best_i


def _make(seg: Segment, words: list, start: float, end: float) -> Segment:
    return Segment(
        start=start,
        end=end,
        text="".join(w.text for w in words).strip(),
        words=list(words),
        speaker=seg.speaker,
        no_speech_prob=seg.no_speech_prob,
        avg_logprob=seg.avg_logprob,
    )
