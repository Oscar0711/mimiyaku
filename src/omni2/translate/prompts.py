"""翻譯用的 prompt。不同模型家族有不同的最佳格式，所以做成可替換的「風格」。

* generic：一般聊天模型（Qwen 等）。system prompt 加上 JSON 批次。
* hy-mt：騰訊 Hy-MT／Hunyuan-MT 的官方格式。沒有 system prompt，用官方的「結構化資料」
  指令翻譯 JSON，術語用「X 翻译成 Y」，前文放在「背景信息」。只在手動指定時使用。
* translategemma：Google TranslateGemma。使用官方固定的 prompt，一次只翻一句，
  沒有放前文的位置。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from omni2.core import lang
from omni2.translate.glossary import Term

# 改了 prompt 就把版本加一，讓舊的翻譯斷點失效、重新翻譯
PROMPT_VERSION = 3


@dataclass
class Request:
    """送給模型的一次請求：聊天格式（messages）或原始文字（prompt）擇一。"""

    messages: list[dict] | None = None
    prompt: str | None = None
    schema: dict | None = None
    sampling: dict = field(default_factory=dict)


def batch_schema(keys: list[str]) -> dict:
    """強制模型對每個編號各輸出一句非空字串，句數一定對得上。"""
    return {
        "type": "object",
        "properties": {k: {"type": "string", "minLength": 1} for k in keys},
        "required": keys,
        "additionalProperties": False,
    }


def parse_json_lines(text: str, keys: list[str]) -> list[str | None] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return [data[k] if isinstance(data.get(k), str) else None for k in keys]


class PromptStyle:
    name = ""
    batch = True  # False：一次只送一句
    temperature: float | None = None  # None：用設定檔的值

    def __init__(self, src_lang: str | None, tgt_lang: str):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def build(self, lines: dict[str, str], *, context: list[tuple[str, str]],
              terms: list[Term], summary: str) -> Request:
        raise NotImplementedError

    def parse(self, text: str, keys: list[str]) -> list[str | None] | None:
        return parse_json_lines(text, keys)


class GenericStyle(PromptStyle):
    name = "generic"

    def system(self) -> str:
        src, tgt = lang.name(self.src_lang), lang.name(self.tgt_lang)
        style = "使用台灣慣用的繁體中文和用語" if self.tgt_lang == "zh-TW" else f"使用自然道地的{tgt}"
        return (
            f"你是專業的影音字幕譯者，負責把{src}字幕翻譯成{tgt}。\n"
            "規則：\n"
            "1. 使用者會給你一個 JSON 物件，鍵是編號、值是一句字幕。請輸出相同編號的 JSON 物件，值換成譯文。\n"
            "2. 一個編號對應一句譯文，不可合併、拆分或省略；只有語氣詞也要翻。\n"
            f"3. 譯文要口語、簡潔、適合當字幕，{style}，保留說話者的語氣。\n"
            "4. 專有名詞依照術語表；術語表沒有的人名用通行譯名，沒有通行譯名就保留原文。\n"
            "5. 原文若明顯是語音辨識錯誤，依上下文推測最合理的意思來翻。\n"
            "6. 「前文」只供參考，不要翻譯也不要輸出。"
        )

    def build(self, lines, *, context, terms, summary) -> Request:
        parts = []
        if summary:
            parts.append(f"【內容摘要】\n{summary}")
        if terms:
            rows = [f"{t.source} → {t.target}" + (f"（{t.note}）" if t.note else "") for t in terms]
            parts.append("【術語表】\n" + "\n".join(rows))
        if context:
            parts.append("【前文】\n" + "\n".join(f"原：{s}\n譯：{t}" for s, t in context))
        parts.append("【待翻譯】\n" + json.dumps(lines, ensure_ascii=False, indent=0))
        messages = [{"role": "system", "content": self.system()},
                    {"role": "user", "content": "\n\n".join(parts)}]
        return Request(messages=messages, schema=batch_schema(list(lines)))


# Hy-MT 官方文件要求語言用中文全名
_HY_NAMES = {"zh": "中文", "zh-CN": "简体中文", "zh-TW": "繁体中文", "zh-HK": "繁体中文",
             "en": "英语", "ja": "日语", "ko": "韩语", "yue": "粤语", "fr": "法语",
             "de": "德语", "es": "西班牙语", "ru": "俄语", "th": "泰语", "vi": "越南语"}


class HyMtStyle(PromptStyle):
    name = "hy-mt"
    temperature = 0.7
    # 官方建議的取樣參數（llama-server 的參數名稱）
    sampling = {"top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05}

    def build(self, lines, *, context, terms, summary) -> Request:
        tgt = _HY_NAMES.get(self.tgt_lang) or _HY_NAMES.get(lang.base(self.tgt_lang), self.tgt_lang)
        parts = []
        if terms:
            parts.append("参考下面的翻译：\n" + "\n".join(f"{t.source} 翻译成 {t.target}" for t in terms))
        background = []
        if summary:
            background.append(summary)
        if context:
            background.append("前文（已翻译）：\n" + "\n".join(f"{s} → {t}" for s, t in context))
        if background:
            parts.append("【背景信息】\n" + "\n\n".join(background))
        parts.append(
            f"# 任务目标\n将下方 JSON 格式数据中的影片字幕翻译为{tgt}，译文要口语、自然。\n\n"
            "# 严格约束\n"
            "1. 结构锁定：绝对保持原有的 JSON 数据结构完全不变，键的数量和顺序都不能改变。\n"
            "2. 选择性翻译：仅翻译值。\n"
            "3. 禁止修改：严禁翻译或更改任何键名。\n\n"
            "# 数据输入\n" + json.dumps(lines, ensure_ascii=False, indent=0)
        )
        return Request(messages=[{"role": "user", "content": "\n\n".join(parts)}],
                       schema=batch_schema(list(lines)), sampling=dict(self.sampling))


_GEMMA_NAMES = {"zh-TW": "Traditional Chinese", "zh-CN": "Simplified Chinese", "zh": "Chinese",
                "en": "English", "ja": "Japanese", "ko": "Korean", "fr": "French", "de": "German",
                "es": "Spanish", "ru": "Russian", "th": "Thai", "vi": "Vietnamese"}


class TranslateGemmaStyle(PromptStyle):
    """TranslateGemma 的官方 prompt。聊天模板要求特殊的 content 結構，所以直接組原始 prompt。"""

    name = "translategemma"
    batch = False
    temperature = 0.0

    def build(self, lines, *, context, terms, summary) -> Request:
        (text,) = lines.values()
        src_code = self.src_lang or "auto"
        src = _GEMMA_NAMES.get(src_code) or _GEMMA_NAMES.get(lang.base(src_code), src_code)
        tgt = _GEMMA_NAMES.get(self.tgt_lang) or _GEMMA_NAMES.get(lang.base(self.tgt_lang), self.tgt_lang)
        user = (
            f"You are a professional {src} ({src_code}) to {tgt} ({self.tgt_lang}) translator. "
            f"Your goal is to accurately convey the meaning and nuances of the original {src} text "
            f"while adhering to {tgt} grammar, vocabulary, and cultural sensitivities.\n"
            f"Produce only the {tgt} translation, without any additional explanations or commentary. "
            f"Please translate the following {src} text into {tgt}:\n\n\n{text}"
        )
        prompt = f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
        return Request(prompt=prompt, sampling={"stop": ["<end_of_turn>"]})

    def parse(self, text, keys):
        return [text.strip() or None]


STYLES = {s.name: s for s in (GenericStyle, HyMtStyle, TranslateGemmaStyle)}


def style_name_for(model: str, override: str = "auto") -> str:
    if override and override != "auto":
        if override not in STYLES:
            raise ValueError(f"未知的 prompt 風格：{override}（可用：auto, {', '.join(STYLES)}）")
        return override
    # Hy-MT 系列刻意不自動選 hy-mt：實測字幕翻譯時官方格式反而比 generic 差
    # （chrF 40.6 對 48.3，同樣溫度 0.2），譯文冗長又到處加語氣詞。要用請在設定裡指定。
    stem = Path(model).name.casefold()
    if "translategemma" in stem:
        return "translategemma"
    return "generic"


def make_style(name: str, src_lang: str | None, tgt_lang: str) -> PromptStyle:
    return STYLES[name](src_lang, tgt_lang)
