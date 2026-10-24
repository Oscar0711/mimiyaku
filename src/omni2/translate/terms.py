"""全文術語先行：翻譯前先讓模型讀完整份逐字稿，整理出摘要和專有名詞的統一譯法。

逐字稿太長時分塊抽取，各塊的術語合併（先出現的優先），各塊的摘要再請模型濃縮成一段。
抽出的結果會先給使用者確認，確認後的術語和摘要才帶進翻譯。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

from omni2.core import lang
from omni2.translate.client import ChatClient
from omni2.translate.glossary import Glossary, Term
from omni2.translate.prompts import Request

log = logging.getLogger(__name__)

TERMS_VERSION = 2
TYPES = ["人名", "暱稱", "地名", "作品", "組織", "遊戲用語", "網路用語", "專業術語", "其他"]

_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "terms": {
            "type": "array",
            "maxItems": 40,
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "minLength": 1},
                    "target": {"type": "string", "minLength": 1},
                    "type": {"type": "string", "enum": TYPES},
                    "note": {"type": "string"},
                },
                "required": ["source", "target", "type", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "terms"],
    "additionalProperties": False,
}

_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


@dataclass
class TermsResult:
    summary: str = ""
    terms: list[dict] = field(default_factory=list)  # 含 type，給使用者看；翻譯時轉成 Glossary

    def glossary(self) -> Glossary:
        return Glossary([Term(t["source"], t["target"], t.get("note", "")) for t in self.terms])

    def to_dict(self) -> dict:
        return {"summary": self.summary, "terms": self.terms}

    @classmethod
    def from_dict(cls, d: dict) -> TermsResult:
        return cls(summary=d.get("summary") or "", terms=list(d.get("terms") or []))


def chunk_lines(lines: list[str], max_chars: int) -> list[str]:
    chunks, cur, size = [], [], 0
    for line in lines:
        if cur and size + len(line) > max_chars:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _prompt(chunk: str, src_lang: str | None, tgt_lang: str, known: list[Term], part: str) -> list[dict]:
    src, tgt = lang.name(src_lang), lang.name(tgt_lang)
    skip = ""
    if known:
        rows = "、".join(f"{t.source} → {t.target}" for t in known)
        skip = f"\n下列詞已經有固定譯法，摘要裡請照用，terms 不用再列：{rows}"
    return [
        {"role": "system", "content": "你是影音字幕翻譯團隊的術語整理員，只輸出 JSON。"},
        {"role": "user", "content": (
            f"以下是一段{src}影片的逐字稿{part}，由語音辨識產生，可能有錯字。請完成兩件事：\n"
            f"1. summary：用{tgt}寫 100 字以內的內容摘要，說明主題、出場人物和情境，"
            "讓譯者知道這是什麼內容。\n"
            "2. terms：列出翻譯時需要統一譯法的專有名詞，例如人名、暱稱、地名、作品名、組織、"
            "遊戲用語、網路用語。source 照逐字稿的寫法；target 是建議的"
            f"{tgt}譯名（台灣慣用說法，人名沒有通行譯名就保留原文）；note 簡短說明這是什麼。"
            "一般詞彙不要列，最多 40 個。" + skip + "\n\n逐字稿：\n" + chunk
        )},
    ]


def extract_terms(
    client: ChatClient,
    lines: list[str],
    *,
    src_lang: str | None,
    tgt_lang: str,
    known: Glossary | None = None,
    chunk_chars: int = 3000,
    progress: Callable[[float], None] | None = None,
) -> TermsResult:
    known = known or Glossary()
    known_sources = set(known.hotwords())
    chunks = chunk_lines(lines, chunk_chars)
    summaries: list[str] = []
    terms: dict[str, dict] = {}
    for i, chunk in enumerate(chunks):
        part = f"（第 {i + 1}／{len(chunks)} 段）" if len(chunks) > 1 else ""
        # 只告訴模型這一段裡有出現的已知詞，內建表很長，全部列出會稀釋指令
        req = Request(messages=_prompt(chunk, src_lang, tgt_lang, known.matches(chunk), part), schema=_SCHEMA)
        result = client.complete(req, temperature=0.2, max_tokens=3072)
        try:
            data = json.loads(result.text)
        except json.JSONDecodeError:
            log.warning("第 %d 段術語抽取的輸出不是合法 JSON，略過", i + 1)
            continue
        if data.get("summary"):
            summaries.append(data["summary"].strip())
        for t in data.get("terms", []):
            source = t["source"].strip()
            # 逐字稿裡根本沒出現的詞（模型自己編的）不收
            if source and source not in terms and source not in known_sources and source in chunk:
                terms[source] = {"source": source, "target": t["target"].strip(),
                                 "type": t.get("type", "其他"), "note": t.get("note", "").strip()}
        if progress:
            progress((i + 1) / (len(chunks) + (1 if len(chunks) > 1 else 0)))

    summary = summaries[0] if summaries else ""
    if len(summaries) > 1:
        req = Request(messages=[
            {"role": "system", "content": "你是影音字幕翻譯團隊的術語整理員，只輸出 JSON。"},
            {"role": "user", "content": (
                f"以下是同一部影片各段的摘要，請合併成一段 150 字以內的{lang.name(tgt_lang)}摘要：\n\n"
                + "\n".join(f"- {s}" for s in summaries)
            )},
        ], schema=_SUMMARY_SCHEMA)
        try:
            summary = json.loads(client.complete(req, temperature=0.2, max_tokens=512).text)["summary"].strip()
        except (json.JSONDecodeError, KeyError):
            summary = " ".join(summaries)
        if progress:
            progress(1.0)
    return TermsResult(summary=summary, terms=list(terms.values()))
