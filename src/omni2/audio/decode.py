"""用 ffmpeg 把任何音檔或影片轉成 16k 單聲道。"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class FfmpegError(RuntimeError):
    pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FfmpegError("找不到 ffmpeg，請安裝後放到 PATH 上")
    return exe


def decode_to_wav(src: Path, dst: Path, sr: int = SAMPLE_RATE, channels: int = 1) -> float:
    """轉成 16-bit PCM wav，回傳長度（秒）。先寫暫存檔，完成後才改名。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.wav")
    cmd = [_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src), "-vn", "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        message = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise FfmpegError(f"ffmpeg 轉檔失敗：{message[-1] if message else proc.returncode}")
    tmp.replace(dst)
    return wav_duration(dst)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path} 不是 16-bit 單聲道 wav")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()
