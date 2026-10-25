"""即時翻譯的紀錄：每次即時翻譯寫成一個 JSONL，結束後可以匯出 SRT、TXT。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from omni2.config.settings import LayoutSettings
from omni2.core import paths
from omni2.core.jsonio import append_jsonl, read_jsonl, write_text_atomic
from omni2.core.types import Segment
from omni2.export import formats as fmt
from omni2.post.layout import build_cues


def sessions_dir() -> Path:
    path = paths.data_dir() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


class SessionLog:
    """第一行是工作階段的資訊，之後每行一句：{"id", "src", "start", "end", "text", "lang", "translation"}。"""

    def __init__(self, info: dict):
        self.started = datetime.now()
        self.path = sessions_dir() / f"{self.started:%Y%m%d-%H%M%S}.jsonl"
        append_jsonl(self.path, [{"started": self.started.isoformat(timespec="seconds"), **info}])

    def add(self, row: dict) -> None:
        append_jsonl(self.path, [row])


def list_sessions() -> list[Path]:
    return sorted(sessions_dir().glob("*.jsonl"), reverse=True)


def load_session(path: Path) -> tuple[dict, list[dict]]:
    rows = read_jsonl(path)
    return (rows[0] if rows else {}), rows[1:]


def export_session(path: Path, out_dir: Path, formats: list[str], layout: LayoutSettings | None = None,
                   bilingual: bool = True) -> list[Path]:
    """把一次即時翻譯匯出成字幕或逐字稿，時間從開始擷取算起。多個來源時標上「我」「對方」。"""
    info, rows = load_session(path)
    layout = layout or LayoutSettings()
    rows.sort(key=lambda r: r["start"])
    labels = info.get("labels", {})
    multi = len({r["src"] for r in rows}) > 1
    segments = [Segment(r["start"], r["end"], r["text"], speaker=labels.get(r["src"], r["src"]) if multi else None)
                for r in rows]
    translations = [r.get("translation") or None for r in rows]
    has_translation = any(translations)
    tag = info.get("tgt_lang", "live") if has_translation else (info.get("src_lang") or "live")
    out = []
    for kind in formats:
        if kind in fmt.SUBTITLE_FORMATS:
            cues = build_cues(segments, translations if has_translation else None, layout, bilingual=bilingual)
            text = {"srt": fmt.to_srt, "vtt": fmt.to_vtt, "ass": fmt.to_ass}[kind](cues, layout)
        elif kind == "txt":
            text = fmt.to_txt(segments, translations if has_translation else None)
        elif kind == "md":
            text = fmt.to_md(segments, translations if has_translation else None, title=path.stem)
        else:
            raise ValueError(f"不支援的格式：{kind}")
        target = out_dir / f"即時翻譯-{path.stem}.{tag}.{kind}"
        write_text_atomic(target, text)
        out.append(target)
    return out


def describe(path: Path) -> str:
    info, rows = load_session(path)
    started = info.get("started", path.stem).replace("T", " ")
    return f"{started}（{len(rows)} 句）"
