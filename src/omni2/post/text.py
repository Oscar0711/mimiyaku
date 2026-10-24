"""字幕文字的寬度計算與斷行。寬度以半形字為單位，全形字算 2。"""

from __future__ import annotations

import math
import unicodedata

# 不能出現在行首、要黏在前一個單位後面的標點
_CLOSING = set("、。，．！？：；…」』）〉》】〕”’ー〜～.,!?:;)]}%")
# 不能出現在行尾、要黏在下一個單位前面的標點
_OPENING = set("「『（〈《【〔“‘([{")


def is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def width(text: str) -> int:
    return sum(2 if is_wide(c) else 1 for c in text)


def atoms(text: str) -> list[str]:
    """切成不可再分的換行單位：英文單字（含後面的空白）或單一全形字，標點黏在相鄰單位上。"""
    out: list[str] = []
    prefix = ""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _OPENING:
            prefix += ch
            i += 1
            continue
        if ch in _CLOSING and out and not prefix:
            out[-1] += ch
            i += 1
            continue
        if ch == " ":
            if out and not prefix:
                out[-1] += ch
            i += 1
            continue
        if is_wide(ch):
            j = i + 1
        else:
            j = i
            while j < n and text[j] != " " and not is_wide(text[j]) and text[j] not in _OPENING:
                j += 1
        out.append(prefix + text[i:j])
        prefix = ""
        i = j
    if prefix:
        out.append(prefix)
    return out


def wrap(text: str, max_width: int) -> list[str]:
    """把一句話斷成寬度相近的幾行。行數不夠時寧可多一行，也不截掉文字。"""
    text = " ".join(text.split())
    if not text:
        return []
    if width(text) <= max_width:
        return [text]
    parts = atoms(text)
    total = width(text)
    n = max(2, math.ceil(total / max_width))
    if n == 2:
        return _best_two_lines(parts, max_width)
    target = total / n
    lines: list[str] = []
    cur, cur_w = "", 0
    for a in parts:
        w = width(a.rstrip())
        if cur and (cur_w + w > max_width or (len(lines) < n - 1 and cur_w + w / 2 > target)):
            lines.append(cur.rstrip())
            cur, cur_w = "", 0
        cur += a
        cur_w += width(a)
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def _best_two_lines(parts: list[str], max_width: int) -> list[str]:
    """最常見的兩行情況：試過每個斷點，取兩行最平均的；一樣平均時讓上行短一點。"""
    best: tuple | None = None
    for i in range(1, len(parts)):
        top = "".join(parts[:i]).rstrip()
        bottom = "".join(parts[i:]).rstrip()
        wt, wb = width(top), width(bottom)
        score = (max(wt, wb) > max_width, max(wt, wb), wt > wb)
        if best is None or score < best[0]:
            best = (score, [top, bottom])
    return best[1] if best else ["".join(parts).rstrip()]


def join_texts(texts: list[str]) -> str:
    """把幾句接成一段：兩邊都是全形字就直接相連，否則補一個空白。"""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if out and not (is_wide(out[-1]) and is_wide(t[0])):
            out += " "
        out += t
    return out
