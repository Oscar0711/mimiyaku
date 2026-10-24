"""把引擎回傳的詞對回原文：補上標點和空白。Qwen3-ForcedAligner、Fun-ASR、Google 都會用到。"""

from __future__ import annotations

import unicodedata

from omni2.core.types import Word

# 開括號屬於下一個詞，不要接在前一個詞後面
_OPENING = set("「『（〈《【〔“‘([{")


def _is_word_char(ch: str) -> bool:
    return ch == "'" or unicodedata.category(ch)[0] in "LN"


def attach_punctuation(text: str, items: list[tuple[str, float, float]]) -> list[Word]:
    """對齊器輸出的詞去掉了標點，這裡依序對回原文，把標點和空白接回每個詞。

    每個詞的文字包含它前面的空白與開括號、後面的標點，串起來就是原文。重新切段要靠
    「。」這類句尾標點，所以不能少。
    """
    words: list[Word] = []
    pos = 0
    folded = text.casefold()
    for token, start, end in items:
        idx = folded.find(token.casefold(), pos) if token else -1
        if idx < 0:
            words.append(Word(start, end, token))
            continue
        stop = idx + len(token)
        while (stop < len(text) and not text[stop].isspace() and not _is_word_char(text[stop])
               and text[stop] not in _OPENING):
            stop += 1
        words.append(Word(start, end, text[pos:stop]))
        pos = stop
    if words and pos < len(text):
        last = words[-1]
        words[-1] = Word(last.start, last.end, last.text + text[pos:])
    return words


def spread_words(text: str, start: float, end: float) -> list[Word]:
    """服務沒有回傳逐詞時間時，依字寬比例估計每個詞的時間，讓過長的句子還能切成字幕大小。"""
    from omni2.post.text import atoms, width

    parts = atoms(text.strip())
    total = sum(max(1, width(p.strip())) for p in parts)
    words, t = [], start
    for p in parts:
        dur = (end - start) * max(1, width(p.strip())) / total if total else 0.0
        words.append(Word(t, t + dur, p))
        t += dur
    return words
