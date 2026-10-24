"""阿里雲 Fun-ASR（DashScope 錄音檔辨識）。

流程：音訊轉成 Opus → 上傳到 DashScope 的暫存空間（oss://，約 48 小時後自動刪除）→
多個檔案送成同一個辨識工作 → 等它完成 → 下載每個檔案的結果 JSON。
金鑰在這個子行程裡從認證管理員讀取，不經過工作參數。
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

import httpx
import numpy as np

from omni2.audio.decode import SAMPLE_RATE
from omni2.audio.encode import encode_opus
from omni2.core import lang
from omni2.core.secrets import require_secret
from omni2.core.types import Segment, Transcript
from omni2.post.text import is_wide
from omni2.stt.base import ProgressFn, SttEngine, TranscribeOptions
from omni2.stt.words import attach_punctuation

log = logging.getLogger(__name__)

REGION_HOSTS = {"intl": "dashscope-intl.aliyuncs.com", "cn": "dashscope.aliyuncs.com"}
_POLL_S = 3.0


def configure_region(region: str) -> None:
    import dashscope

    host = REGION_HOSTS.get(region)
    if host is None:
        raise ValueError(f"未知的 DashScope 區域：{region}（可用：{', '.join(REGION_HOSTS)}）")
    dashscope.base_http_api_url = f"https://{host}/api/v1"
    dashscope.base_websocket_api_url = f"wss://{host}/api-ws/v1/inference"


def check_key(api_key: str, region: str, model: str = "fun-asr") -> tuple[bool, str]:
    """用「取得上傳憑證」這個不花錢的請求確認金鑰和區域。回傳（是否成功, 說明）。"""
    from dashscope.utils.oss_utils import OssUtils

    configure_region(region)
    try:
        resp = OssUtils.get_upload_certificate(model=model, api_key=api_key)
    except Exception as e:  # noqa: BLE001 — 網路錯誤等
        return False, f"{type(e).__name__}: {e}"
    if resp.status_code == 200:
        return True, "金鑰有效"
    return False, f"{resp.code}: {resp.message}"


_LONE_I = re.compile(r"\bi\b")


def tidy_text(text: str, language: str | None) -> str:
    """修正 Fun-ASR 輸出的小毛病（實測）：
    * 日文、中文在停頓處插入空格（「このゲーム ボスが」）：刪掉夾在兩個全形字之間的空格。
    * 英文句首和單獨的 i 常是小寫（「today i want…」）：改成大寫。
    韓文本來就用空格分詞，不處理（它偶爾會在詞中間多空格，但無法用規則判斷）。
    """
    base = lang.base(language)
    if base in ("ja", "zh", "yue"):
        chars = list(text)
        text = "".join(c for i, c in enumerate(chars)
                       if not (c == " " and 0 < i < len(chars) - 1
                               and is_wide(chars[i - 1]) and is_wide(chars[i + 1])))
    elif base == "en":
        text = _LONE_I.sub("I", text)
        for i, c in enumerate(text):
            if c.isalpha():
                text = text[:i] + c.upper() + text[i + 1:]
                break
    return text


def parse_result(data: dict, language: str | None = None) -> list[Segment]:
    """DashScope 錄音檔辨識的結果 JSON → 片段（含逐詞時間，標點接回詞上）。
    language 沒給時依文字內容判斷，用來決定 tidy_text 怎麼修。"""
    sentences = [s for t in data.get("transcripts", []) for s in t.get("sentences", [])]
    if language is None:
        language = lang.guess_from_text("".join(s.get("text") or "" for s in sentences))
    segments = []
    for sentence in sentences:
        text = tidy_text((sentence.get("text") or "").strip(), language)
        if not text:
            continue
        items = [(w.get("text", ""), w["begin_time"] / 1000, w["end_time"] / 1000)
                 for w in sentence.get("words", []) if "begin_time" in w]
        segments.append(Segment(
            start=sentence["begin_time"] / 1000,
            end=sentence["end_time"] / 1000,
            text=text,
            words=attach_punctuation(text, items) if items else [],
        ))
    segments.sort(key=lambda s: s.start)
    return segments


class FunAsrEngine(SttEngine):
    name = "funasr"
    batches_files = True  # 多個檔案送成同一個辨識工作

    def load(self) -> None:
        cloud = self.settings.cloud
        self.api_key = require_secret("dashscope", "阿里雲 DashScope")
        self.model = cloud.funasr_model
        configure_region(cloud.dashscope_region)

    def transcribe(self, audio: np.ndarray, opts: TranscribeOptions,
                   progress: ProgressFn | None = None) -> Transcript:
        cb = (lambda _i, p: progress(p)) if progress else None
        return self.transcribe_many([audio], opts, cb)[0]

    def transcribe_many(self, audios: list[np.ndarray], opts: TranscribeOptions,
                        progress: Callable[[int, float], None] | None = None) -> list[Transcript]:
        from dashscope.audio.asr import Transcription
        from dashscope.utils.oss_utils import OssUtils

        report = progress or (lambda _i, _p: None)
        urls: list[str] = []
        names: list[str] = []  # 每個檔案獨一無二的檔名，用來把結果對回檔案
        certificate = None
        with tempfile.TemporaryDirectory(prefix="omni2-funasr-") as tmp:
            for i, audio in enumerate(audios):
                names.append(f"omni2-{uuid.uuid4().hex}.ogg")
                path = encode_opus(audio, Path(tmp) / names[-1])
                url, certificate = OssUtils.upload(model=self.model, file_path=str(path), api_key=self.api_key,
                                                   upload_certificate=certificate)
                urls.append(url)
                report(i, 0.3)
                log.info("已上傳 %s（%.1f MB）", path.name, path.stat().st_size / 1e6)

        kwargs = {"headers": {"X-DashScope-OssResourceResolve": "enable"}}
        if opts.language:
            kwargs["language_hints"] = [lang.base(opts.language)]
        task = Transcription.async_call(model=self.model, file_urls=urls, api_key=self.api_key, **kwargs)
        if task.status_code != 200:
            raise RuntimeError(f"Fun-ASR 送出工作失敗：{task.code}: {task.message}")
        task_id = task.output["task_id"]
        log.info("Fun-ASR 工作 %s 已送出（%d 個檔案）", task_id, len(urls))
        while True:
            resp = Transcription.fetch(task=task_id, api_key=self.api_key)
            if resp.status_code != 200:
                raise RuntimeError(f"Fun-ASR 查詢失敗：{resp.code}: {resp.message}")
            status = resp.output["task_status"]
            if status in ("SUCCEEDED", "FAILED"):
                break
            for i in range(len(audios)):
                report(i, 0.5)
            time.sleep(_POLL_S)

        results = resp.output.get("results") or []
        out = []
        for i, (audio, name) in enumerate(zip(audios, names)):
            # 回傳的 file_url 不是送出的 oss:// 網址，順序也不保證一樣，所以用檔名對回去
            # （實測按順序對應時，三個檔案的結果全部錯開）
            item = next((r for r in results if name in (r.get("file_url") or "")), None)
            if item is None:
                raise RuntimeError(f"Fun-ASR 的結果裡找不到第 {i + 1} 個檔案（{name}）")
            if item.get("subtask_status") != "SUCCEEDED":
                raise RuntimeError(f"Fun-ASR 辨識失敗：{item.get('code')}: {item.get('message')}")
            data = httpx.get(item["transcription_url"], timeout=120).json()
            # Fun-ASR 不回傳語言，依文字內容判斷
            language = opts.language or lang.guess_from_text(
                "".join(x.get("text") or "" for t in data.get("transcripts", []) for x in t.get("sentences", [])))
            segments = parse_result(data, language)
            out.append(Transcript(segments=segments, language=language, duration=len(audio) / SAMPLE_RATE))
            report(i, 1.0)
        return out
