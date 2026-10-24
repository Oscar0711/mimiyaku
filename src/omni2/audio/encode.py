"""把 16k 單聲道音訊編碼成要上傳給雲端服務的格式。"""

from __future__ import annotations

import io
import subprocess
import wave
from pathlib import Path

import numpy as np

from omni2.audio.decode import SAMPLE_RATE, FfmpegError, _ffmpeg

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


def wav_bytes(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """一小段音訊包成 wav（給逐段送出的 API）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(to_pcm16(audio))
    return buf.getvalue()


def encode_opus(audio: np.ndarray, dst: Path, bitrate: str = "32k") -> Path:
    """整段音訊轉成 Opus（.ogg）。一小時約 15 MB，原始 wav 則超過 100 MB，上傳快很多。"""
    cmd = [_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
           "-c:a", "libopus", "-b:a", bitrate, "-application", "voip", str(dst)]
    proc = subprocess.run(cmd, input=to_pcm16(audio), capture_output=True, creationflags=_NO_WINDOW)
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise FfmpegError(f"Opus 編碼失敗：{message[-1] if message else proc.returncode}")
    return dst
