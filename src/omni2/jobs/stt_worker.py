"""STT 階段，在獨立子行程裡執行（見 omni2.workers.stage）。

模型只載入一次。能跨檔案批次推論的引擎（Qwen3-ASR）會把多個檔案一起送；其他引擎逐檔
處理。每完成一個檔案就寫出 transcript.json，中途停止的話已完成的檔案不必重做。
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from omni2.audio.decode import read_wav
from omni2.config.settings import SttSettings
from omni2.core.jsonio import write_json_atomic
from omni2.core.types import Transcript
from omni2.post.resegment import split_long
from omni2.stt.base import TranscribeOptions
from omni2.stt.hallucination import filter_segments
from omni2.stt.registry import create_engine
from omni2.vad.presets import get_preset

log = logging.getLogger(__name__)

# 一起批次推論的檔案總長上限（秒），避免一次把太多音訊載進記憶體
_GROUP_SECONDS = 2 * 3600


def _groups(items: list[dict], batches_files: bool) -> list[list[dict]]:
    """同一組的檔案必須共用語言、VAD 和提示詞，才能放進同一次推論。"""
    if not batches_files:
        return [[item] for item in items]
    groups: list[list[dict]] = []
    total = 0.0
    for item in items:
        key = (item["language"], item["vad"], tuple(item["hotwords"]))
        if groups and total + item["duration"] <= _GROUP_SECONDS and groups[-1][0]["_key"] == key:
            groups[-1].append(item)
            total += item["duration"]
        else:
            groups.append([item])
            total = item["duration"]
        item["_key"] = key
    return groups


def _finish(item: dict, transcript: Transcript, settings: SttSettings, emit) -> None:
    segments = transcript.segments
    dropped = []
    if settings.filter_hallucinations:
        segments, dropped = filter_segments(segments)
        for seg, reason in dropped:
            log.info("略過 %.1fs「%s」：%s", seg.start, seg.text, reason)
    transcript.segments = split_long(segments, item["max_duration"], item["max_width"])
    data = transcript.to_dict()
    data["fp"] = item["fp"]
    write_json_atomic(Path(item["out"]), data)
    emit({"type": "job_done", "job": item["job"], "language": transcript.language,
          "segments": len(transcript.segments), "dropped": len(dropped)})


def run(payload: dict, emit) -> dict:
    settings = SttSettings(**payload["stt"])
    engine = create_engine(settings)
    emit({"type": "status", "message": f"載入 {settings.engine}（{settings.active_model()}）"})
    engine.load()

    for group in _groups(payload["items"], engine.batches_files):
        last: dict[str, float] = {}

        def progress(k: int, p: float, group=group) -> None:
            job = group[k]["job"]
            if p - last.get(job, -1.0) >= 0.01 or p >= 1.0:
                last[job] = p
                emit({"type": "progress", "job": job, "value": p})

        try:
            for item in group:
                emit({"type": "job_start", "job": item["job"]})
            audios = [read_wav(Path(item["audio"])) for item in group]
            first = group[0]
            opts = TranscribeOptions(
                language=first["language"],
                vad=get_preset(first["vad"]),
                hotwords=first["hotwords"],
                beam_size=settings.beam_size,
            )
            transcripts = engine.transcribe_many(audios, opts, progress)
        except Exception as e:  # noqa: BLE001 — 這一組失敗不影響其他組
            for item in group:
                emit({"type": "job_error", "job": item["job"], "message": f"{type(e).__name__}: {e}",
                      "traceback": traceback.format_exc()})
            continue
        for item, transcript in zip(group, transcripts):
            try:
                _finish(item, transcript, settings, emit)
            except Exception as e:  # noqa: BLE001
                emit({"type": "job_error", "job": item["job"], "message": f"{type(e).__name__}: {e}",
                      "traceback": traceback.format_exc()})
    engine.close()
    return {"ok": True}
