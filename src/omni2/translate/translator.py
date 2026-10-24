"""批次翻譯：每批 N 句一起送，帶前文與術語，逐句檢查品質，不合格的單獨重試。"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

from omni2.core import lang
from omni2.translate import prompts, quality
from omni2.translate.client import ChatClient
from omni2.translate.glossary import Glossary
from omni2.workers.stage import Cancelled

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Issue:
    index: int
    reason: str


def _key(text: str) -> str:
    return " ".join(text.split()).casefold()


# 純簡體、但因為歷史原因也收在 Big5 裡的常見字
_SIMPLIFIED_IN_BIG5 = set("网气么机体价优儿几种极构确离胜听惊怀坏复万与党圣并洁洒虫赶挂据扰愿异柜触")


def _in_big5(ch: str) -> bool:
    try:
        ch.encode("cp950")
        return True
    except UnicodeEncodeError:
        return False


def make_converter(config: str, tgt_lang: str) -> Callable[[str], str]:
    """目標是繁中時，把模型偶爾冒出的簡體字轉掉。

    不能無條件套 OpenCC：它會把本來就正確的繁體改壞（實測「濃郁」→「濃鬱」，因為「郁」
    在簡體裡對應多個繁體字）。所以只處理「確定是簡體」的字，也就是 s2t 會改、而且不在
    Big5 繁體字集裡的字。整句都是簡體時才交給 OpenCC 做詞組轉換（软件→軟體）。
    """
    if not config or tgt_lang not in ("zh-TW", "zh-HK", "zh-Hant"):
        return lambda s: s
    from opencc import OpenCC

    phrases, chars, to_simplified = OpenCC(config), OpenCC("s2t"), OpenCC("t2s")

    def is_simplified(ch: str) -> bool:
        return (not _in_big5(ch) or ch in _SIMPLIFIED_IN_BIG5) and chars.convert(ch) != ch

    def convert(text: str) -> str:
        if not any(is_simplified(ch) for ch in text):
            return text
        if to_simplified.convert(text) == text:  # 沒有任何繁體專用字：整句是簡體
            return phrases.convert(text)
        return "".join(chars.convert(ch) if is_simplified(ch) else ch for ch in text)

    return convert


_QUOTE_PAIRS = (("「", "」"), ("『", "』"), ("“", "”"), ("（", "）"))


def tidy(text: str, source: str) -> str:
    """去掉模型多吐出來、原文沒有的不成對引號（實測 Hy-MT2 偶爾在句尾多一個「」」）。"""
    text = text.strip()
    for open_, close in _QUOTE_PAIRS:
        if text.count(open_) == text.count(close) or source.count(open_) != source.count(close):
            continue
        if text.endswith(close) and text.count(close) > text.count(open_):
            text = text[: -len(close)].rstrip()
        elif text.startswith(open_) and text.count(open_) > text.count(close):
            text = text[len(open_):].lstrip()
    return text


class BatchTranslator:
    def __init__(
        self,
        client: ChatClient,
        *,
        src_lang: str | None,
        tgt_lang: str,
        glossary: Glossary | None = None,
        summary: str = "",
        batch_lines: int = 10,
        context_lines: int = 3,
        temperature: float = 0.2,
        max_retries: int = 2,
        opencc: str = "s2twp",
        style: str = "generic",
    ):
        self.client = client
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.glossary = glossary or Glossary()
        self.summary = summary
        self.style = prompts.make_style(style, src_lang, tgt_lang)
        self.batch_lines = max(1, batch_lines) if self.style.batch else 1
        self.context_lines = context_lines
        self.temperature = temperature if self.style.temperature is None else self.style.temperature
        self.max_retries = max_retries
        self.convert = make_converter(opencc, tgt_lang)
        self.issues: list[Issue] = []

    def translate_all(
        self,
        texts: list[str],
        *,
        done: dict[int, str] | None = None,
        on_result: Callable[[dict[int, str]], None] | None = None,
        progress: Callable[[float], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> list[str]:
        """done 是上次已完成的部分（斷點續作）。每完成一批就呼叫 on_result，讓呼叫端寫斷點。"""
        n = len(texts)
        results: dict[int, str] = dict(done or {})
        if lang.same_language(self.src_lang, self.tgt_lang):
            out = {i: self.convert(texts[i]) for i in range(n) if i not in results}
            results.update(out)
            if on_result and out:
                on_result(out)
            return [results[i] for i in range(n)]

        cache = {_key(texts[i]): results[i] for i in results}
        pending = [i for i in range(n) if i not in results]
        for pos in range(0, len(pending), self.batch_lines):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            batch = pending[pos : pos + self.batch_lines]
            out: dict[int, str] = {}
            todo: list[int] = []
            seen: set[str] = set()
            for i in batch:
                k = _key(texts[i])
                if not k:
                    out[i] = ""
                elif k in cache:
                    out[i] = cache[k]
                elif k not in seen:
                    seen.add(k)
                    todo.append(i)
            if todo:
                translated = self._translate(texts, todo, results)
                for i, t in translated.items():
                    cache[_key(texts[i])] = t
                out.update(translated)
            for i in batch:  # 同一批裡重複的句子
                if i not in out:
                    out[i] = cache[_key(texts[i])]
            results.update(out)
            if on_result:
                on_result(out)
            if progress:
                progress(len(results) / n)
        return [results[i] for i in range(n)]

    # ---- 內部 ----

    def _translate(self, texts: list[str], idx: list[int], known: dict[int, str]) -> dict[int, str]:
        raw = self._call([texts[i] for i in idx], self._context(texts, idx[0], known), self.temperature)
        if raw is None and len(idx) > 1:
            # 整批失敗（被截斷或格式錯）：對半切開重來
            half = len(idx) // 2
            first = self._translate(texts, idx[:half], known)
            second = self._translate(texts, idx[half:], {**known, **first})
            return {**first, **second}
        raw = raw or [None] * len(idx)
        out: dict[int, str] = {}
        for i, t in zip(idx, raw):
            out[i] = self._check_or_retry(texts, i, t, {**known, **out})
        return out

    def _check_or_retry(self, texts: list[str], i: int, text: str | None, known: dict[int, str]) -> str:
        best = text
        reason = quality.problem(texts[i], text, self.src_lang, self.tgt_lang)
        attempt = 0
        while reason and attempt < self.max_retries:
            attempt += 1
            temperature = min(1.0, self.temperature + 0.3 * attempt)
            retry = self._call([texts[i]], self._context(texts, i, known), temperature)
            candidate = retry[0] if retry else None
            new_reason = quality.problem(texts[i], candidate, self.src_lang, self.tgt_lang)
            if not new_reason or not best:
                best = candidate
            reason = new_reason
        if reason:
            self.issues.append(Issue(i, reason))
            log.warning("第 %d 句翻譯未通過檢查（%s）：%s → %s", i + 1, reason, texts[i], best)
        return self.convert(tidy(best or "", texts[i]))

    def _context(self, texts: list[str], first: int, known: dict[int, str]) -> list[tuple[str, str]]:
        ctx: list[tuple[str, str]] = []
        j = first - 1
        while j >= 0 and len(ctx) < self.context_lines:
            if known.get(j):
                ctx.append((texts[j], known[j]))
            j -= 1
        return ctx[::-1]

    def _call(self, lines: list[str], context: list[tuple[str, str]], temperature: float) -> list[str | None] | None:
        """回傳和 lines 對齊的譯文；整批不可用時回傳 None。連線錯誤直接往上丟。"""
        keys = [str(k + 1) for k in range(len(lines))]
        req = self.style.build(
            dict(zip(keys, lines)),
            context=context,
            terms=self.glossary.matches(" ".join(lines)),
            summary=self.summary,
        )
        max_tokens = min(4096, 64 + sum(len(line) * 4 + 24 for line in lines))
        result = self.client.complete(req, temperature=temperature, max_tokens=max_tokens)
        if result.finish_reason == "length":
            log.info("輸出被 token 上限截斷（%d 句），改成分批重試", len(lines))
            return None
        parsed = self.style.parse(result.text, keys)
        if parsed is None:
            log.info("輸出格式不對（%d 句）", len(lines))
        return parsed
