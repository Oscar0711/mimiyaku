"""即時翻譯：一句確定的原文 → 用 llama-server 串流翻譯，譯文逐字送出。

和檔案模式不同，這裡一次只翻一句、要越快越好，所以不用 JSON 批次，直接輸出純文字；
前幾句的原文和譯文當上下文，術語表照樣帶入。重複的句子直接用快取，來源和目標語言相同就跳過。
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from omni2.core import lang
from omni2.translate.client import ChatClient
from omni2.translate.glossary import Glossary
from omni2.translate.translator import make_converter, tidy

log = logging.getLogger(__name__)

EventFn = Callable[[dict], None]


def system_prompt(src_lang: str | None, tgt_lang: str) -> str:
    src, tgt = lang.name(src_lang), lang.name(tgt_lang)
    style = "使用台灣慣用的繁體中文和用語" if tgt_lang == "zh-TW" else f"使用自然道地的{tgt}"
    return (
        f"你是即時字幕譯者，把{src}翻譯成{tgt}。\n"
        "規則：\n"
        "1. 只輸出這一句的譯文，不要加引號、說明、原文或拼音。\n"
        f"2. 譯文要口語、簡潔，{style}，保留說話者的語氣。\n"
        "3. 專有名詞依照術語表。\n"
        "4. 原文若明顯是語音辨識錯誤，依上下文推測最合理的意思來翻。\n"
        "5. 「前文」只供參考，不要翻譯也不要輸出。"
    )


class LiveTranslator:
    def __init__(self, client: ChatClient, tgt_lang: str, glossary_for: Callable[[str | None], Glossary],
                 context_lines: int = 3, opencc: str = "s2twp", temperature: float = 0.2):
        self.client = client
        self.tgt_lang = tgt_lang
        self.glossary_for = glossary_for
        self.context_lines = context_lines
        self.temperature = temperature
        self.convert = make_converter(opencc, tgt_lang)
        self.cache: dict[str, str] = {}
        self.history: list[tuple[str, str]] = []

    def warm_up(self) -> None:
        """第一次請求要建立 prompt cache，比較慢；開始收音前先做一次。"""
        for _ in self.client.stream([{"role": "system", "content": system_prompt("ja", self.tgt_lang)},
                                     {"role": "user", "content": "【待翻譯】\nこんにちは"}], max_tokens=8):
            pass

    def translate(self, event: dict, emit: EventFn) -> str:
        """翻譯一句確定的原文。過程中送出 {"type": "translation", "id", "text", "done"}。"""
        text, src_lang = event["text"].strip(), event.get("lang")
        key = " ".join(text.split()).casefold()
        if not text:
            return ""
        if lang.same_language(src_lang, self.tgt_lang):
            result = self.convert(text)
        elif key in self.cache:
            result = self.cache[key]
        else:
            result = self._stream(text, src_lang, event, emit)
            self.cache[key] = result
        self.history = (self.history + [(text, result)])[-self.context_lines:]
        emit({"type": "translation", "id": event["id"], "text": result, "done": True})
        return result

    def _stream(self, text: str, src_lang: str | None, event: dict, emit: EventFn) -> str:
        parts = []
        terms = self.glossary_for(src_lang).matches(text)
        if terms:
            parts.append("【術語表】\n" + "\n".join(f"{t.source} → {t.target}" for t in terms))
        if self.history:
            parts.append("【前文】\n" + "\n".join(f"原：{s}\n譯：{t}" for s, t in self.history))
        parts.append(f"【待翻譯】\n{text}")
        messages = [{"role": "system", "content": system_prompt(src_lang, self.tgt_lang)},
                    {"role": "user", "content": "\n\n".join(parts)}]
        acc, first = "", None
        for delta in self.client.stream(messages, temperature=self.temperature, max_tokens=64 + len(text) * 4):
            if first is None:
                first = time.monotonic()
                emit({"type": "first_token", "id": event["id"], "at": first})
            acc += delta
            # 模型偶爾會在譯文後面接著換行寫說明：只留第一行
            shown = acc.strip().split("\n")[0]
            emit({"type": "translation", "id": event["id"], "text": self.convert(shown), "done": False})
        return self.convert(tidy(acc.strip().split("\n")[0], text))
