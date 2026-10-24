"""術語表：固定專有名詞的譯法，原文同時當作 STT 的辨識提示。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

import yaml

from omni2.core import lang
from omni2.core.jsonio import write_text_atomic

_ASCII_WORD = re.compile(r"^[\w .'-]+$", re.ASCII)


@dataclass(frozen=True, slots=True)
class Term:
    source: str
    target: str
    note: str = ""


def _terms_from(data: list | None) -> list[Term]:
    return [
        Term(str(d["source"]).strip(), str(d.get("target") or "").strip(), str(d.get("note") or "").strip())
        for d in (data or [])
        if isinstance(d, dict) and d.get("source")
    ]


class Glossary:
    def __init__(self, terms: list[Term] | None = None):
        self.terms = list(terms or [])
        self._patterns: dict[str, re.Pattern] = {}

    @classmethod
    def load(cls, path: Path | None) -> Glossary:
        if not path or not path.exists():
            return cls()
        return cls(_terms_from(yaml.safe_load(path.read_text(encoding="utf-8"))))

    @classmethod
    def from_dicts(cls, data: list[dict]) -> Glossary:
        return cls(_terms_from(data))

    @classmethod
    def builtin(cls, name: str, src_lang: str | None) -> Glossary:
        """內建術語表（omni2/data/glossary/<name>.<語言>.yaml），沒有這個語言就是空的。"""
        filename = f"{name}.{lang.base(src_lang)}.yaml"
        ref = resources.files("omni2.data.glossary").joinpath(filename)
        if not ref.is_file():
            return cls()
        return cls(_terms_from(yaml.safe_load(ref.read_text(encoding="utf-8"))))

    def save(self, path: Path) -> None:
        text = yaml.safe_dump([asdict(t) for t in self.terms], allow_unicode=True, sort_keys=False)
        write_text_atomic(path, text)

    def merged(self, other: Glossary) -> Glossary:
        """other 的條目優先（例如這份檔案專屬的術語蓋過全域術語）。"""
        by_source = {t.source: t for t in self.terms}
        by_source.update({t.source: t for t in other.terms})
        return Glossary(list(by_source.values()))

    def _matches_one(self, term: Term, text: str, folded: str) -> bool:
        if _ASCII_WORD.match(term.source):
            # 英文要整詞比對，否則「mod」會誤中「model」
            pattern = self._patterns.get(term.source)
            if pattern is None:
                pattern = re.compile(rf"\b{re.escape(term.source)}\b", re.IGNORECASE)
                self._patterns[term.source] = pattern
            return bool(pattern.search(text))
        return term.source.casefold() in folded

    def matches(self, text: str) -> list[Term]:
        folded = text.casefold()
        return [t for t in self.terms if t.target and self._matches_one(t, text, folded)]

    def hotwords(self) -> list[str]:
        return [t.source for t in self.terms]

    def digest(self) -> str:
        payload = json.dumps([asdict(t) for t in self.terms], ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.terms)
