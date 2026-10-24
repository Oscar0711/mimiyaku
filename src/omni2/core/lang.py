"""語言碼。內部一律用 BCP-47 風格（ja、en、ko、zh-TW），送進 STT 時再轉成引擎要的格式。"""

from __future__ import annotations

NAMES = {
    "ja": "日文",
    "en": "英文",
    "ko": "韓文",
    "zh": "中文",
    "zh-TW": "繁體中文（台灣）",
    "zh-CN": "簡體中文",
    "yue": "粵語",
    "fr": "法文",
    "de": "德文",
    "es": "西班牙文",
    "ru": "俄文",
    "th": "泰文",
    "vi": "越南文",
}


def base(lang: str | None) -> str:
    return (lang or "").split("-")[0].lower()


def name(lang: str | None) -> str:
    if not lang:
        return "原文"
    return NAMES.get(lang) or NAMES.get(base(lang)) or lang


def stt_code(lang: str | None) -> str | None:
    """Whisper 類引擎只認主語言碼。"""
    return base(lang) or None


def same_language(src: str | None, tgt: str | None) -> bool:
    return bool(src) and base(src) == base(tgt)


def guess_from_text(text: str) -> str | None:
    """不回傳語言的雲端服務，依文字內容粗略判斷（只用來選翻譯方向和內建術語表）。"""
    if not text:
        return None
    kana = sum("぀" <= c <= "ヿ" for c in text)
    hangul = sum("가" <= c <= "힯" for c in text)
    cjk = sum("一" <= c <= "鿿" for c in text)
    latin = sum(c.isascii() and c.isalpha() for c in text)
    if kana > 0.05 * len(text):
        return "ja"
    if hangul > 0.2 * len(text):
        return "ko"
    if cjk > latin:
        return "zh"
    return "en" if latin else None
