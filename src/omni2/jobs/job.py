"""一個輸入檔＝一個 job。每個 job 有自己的資料夾，存各階段的產物和斷點。

斷點的有效性用「指紋」判斷：每個階段的指紋＝上游指紋＋這個階段用到的設定。改了某個
設定，只有受影響的階段和它之後的階段會重做，其餘直接沿用。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from omni2.core.jsonio import append_jsonl, read_json, read_jsonl, write_json_atomic
from omni2.core.types import Transcript


@dataclass
class JobSpec:
    input: str
    src_lang: str | None = None
    translate: bool = True
    tgt_lang: str = "zh-TW"
    formats: list[str] = field(default_factory=lambda: ["srt"])
    bilingual: bool = False
    out_dir: str | None = None
    separate_vocals: bool = False
    diarize: bool = False
    num_speakers: int | None = None  # None＝自動判斷
    terms_first: bool = False


def is_url(text: str) -> bool:
    return text.startswith(("http://", "https://"))


_BAD_CHARS = str.maketrans({c: "_" for c in r'<>:"/\|?*'})


def safe_filename(name: str) -> str:
    return name.translate(_BAD_CHARS).strip(" .")[:150] or "untitled"


def fingerprint(*parts) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def job_id(input: str) -> str:
    """本地檔用「路徑＋大小＋修改時間」，檔案換了內容就是新的 job。"""
    if is_url(input):
        return fingerprint("url", input)
    p = Path(input).resolve()
    st = p.stat()
    return fingerprint("file", str(p).casefold(), st.st_size, st.st_mtime_ns)


class Job:
    def __init__(self, spec: JobSpec, root: Path):
        self.spec = spec
        self.id = job_id(spec.input)
        self.dir = root / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state: dict = read_json(self.state_path, {}) or {}
        self.state.setdefault("input", spec.input)
        self.state.setdefault("stages", {})
        self.error: str | None = None
        self.outputs: list[Path] = []

    @property
    def name(self) -> str:
        if is_url(self.spec.input):
            return safe_filename(self.state.get("title") or self.id)
        return Path(self.spec.input).stem

    @property
    def source_path(self) -> Path:
        """原始音訊：本地檔就是輸入檔，網址則是下載下來的檔案。"""
        if is_url(self.spec.input):
            return Path(self.state["source"])
        return Path(self.spec.input)

    @property
    def state_path(self) -> Path:
        return self.dir / "job.json"

    @property
    def audio_path(self) -> Path:
        return self.dir / "audio.wav"

    @property
    def transcript_path(self) -> Path:
        return self.dir / "transcript.json"

    @property
    def translation_path(self) -> Path:
        return self.dir / "translation.jsonl"

    @property
    def vocals_path(self) -> Path:
        return self.dir / "vocals.wav"

    @property
    def diarized_path(self) -> Path:
        return self.dir / "diarized.json"

    @property
    def terms_path(self) -> Path:
        return self.dir / "terms.json"

    @property
    def stt_audio_path(self) -> Path:
        return self.vocals_path if self.spec.separate_vocals else self.audio_path

    def load_diarized(self, fp: str) -> Transcript | None:
        """講者分離後的片段（在換人的地方切開過），格式和 transcript.json 相同。"""
        data = read_json(self.diarized_path)
        if not data or data.get("fp") != fp:
            return None
        return Transcript.from_dict(data)

    def load_terms(self, fp: str) -> dict | None:
        data = read_json(self.terms_path)
        if not data or data.get("fp") != fp:
            return None
        return data

    def save_terms(self, fp: str, data: dict, reviewed: bool) -> None:
        write_json_atomic(self.terms_path, {"fp": fp, "reviewed": reviewed, **data})

    def save(self) -> None:
        write_json_atomic(self.state_path, self.state)

    def stage(self, name: str) -> dict:
        return self.state["stages"].get(name, {})

    def stage_done(self, name: str, fp: str) -> bool:
        st = self.stage(name)
        return st.get("status") == "done" and st.get("fp") == fp

    def set_stage(self, name: str, fp: str, status: str, **info) -> None:
        self.state["stages"][name] = {"fp": fp, "status": status, **info}
        self.save()

    # ---- 階段產物 ----

    def load_transcript(self, fp: str | None = None) -> Transcript | None:
        data = read_json(self.transcript_path)
        if not data or (fp is not None and data.get("fp") != fp):
            return None
        return Transcript.from_dict(data)

    def load_translations(self, fp: str) -> dict[int, str]:
        """讀翻譯斷點。指紋對不上（設定改過）就整份作廢。"""
        rows = read_jsonl(self.translation_path)
        if not rows or rows[0].get("fp") != fp:
            self.translation_path.unlink(missing_ok=True)
            append_jsonl(self.translation_path, [{"fp": fp}])
            return {}
        return {row["i"]: row["t"] for row in rows[1:] if "i" in row}

    def append_translations(self, out: dict[int, str]) -> None:
        append_jsonl(self.translation_path, [{"i": i, "t": t} for i, t in sorted(out.items())])
