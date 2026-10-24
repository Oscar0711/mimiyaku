"""譯文品質檢查。只擋明顯錯誤；有疑問時寧可放行，避免無謂的重試。"""

from __future__ import annotations

import re

from omni2.core import lang
from omni2.stt.hallucination import compression_ratio

_KANA = re.compile(r"[぀-ゟ゠-ヿ]")
_HANGUL = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")
_LETTERS = re.compile(r"[぀-ヿ가-힯A-Za-z]")
_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def _norm(text: str) -> str:
    return _PUNCT.sub("", text.casefold())


def problem(source: str, translation: str | None, src_lang: str | None, tgt_lang: str) -> str | None:
    if translation is None:
        return "沒有輸出"
    t = translation.strip()
    if not t:
        return "空白"
    if lang.base(src_lang) != lang.base(tgt_lang):
        # 照抄原文：原文要有足夠的假名、韓文或拉丁字母才算（全漢字的日文名詞照抄是正常的）
        if _norm(t) == _norm(source) and len(_LETTERS.findall(source)) >= 4:
            return "照抄原文"
    if lang.base(tgt_lang) == "zh":
        if len(_KANA.findall(t)) >= 2:
            return "夾雜日文假名"
        if _HANGUL.search(t):
            return "夾雜韓文"
    if len(t) >= 20 and compression_ratio(t) > 3.0:
        return "異常重複"
    if len(t) > max(60, 6 * len(source)):
        return "譯文異常冗長"
    return None
