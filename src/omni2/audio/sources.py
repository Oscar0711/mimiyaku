"""即時模式的音訊來源：麥克風、系統聲音（WASAPI loopback）、指定程式（process loopback）。

每個來源都以 16 kHz 單聲道 float32、約 0.1 秒一塊的方式回呼（在擷取的執行緒裡）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
AudioFn = Callable[[np.ndarray], None]


@dataclass
class Device:
    index: int
    name: str
    rate: int
    channels: int


@dataclass
class AudioApp:
    pid: int
    name: str


def list_microphones() -> list[Device]:
    """WASAPI 的輸入裝置。第一個是預設麥克風。"""
    import pyaudiowpatch as pa

    p = pa.PyAudio()
    try:
        info = p.get_host_api_info_by_type(pa.paWASAPI)
        devices = [Device(d["index"], d["name"], int(d["defaultSampleRate"]), int(d["maxInputChannels"]))
                   for d in (p.get_device_info_by_index(i) for i in range(p.get_device_count()))
                   if d["hostApi"] == info["index"] and d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice")]
        devices.sort(key=lambda d: d.index != info["defaultInputDevice"])  # 預設麥克風排第一
        return devices
    finally:
        p.terminate()


def list_outputs() -> list[Device]:
    """可以擷取的輸出裝置（它們的 loopback 版本）。第一個是預設輸出。"""
    import pyaudiowpatch as pa

    p = pa.PyAudio()
    try:
        default = p.get_default_wasapi_loopback()
        devices = [Device(d["index"], d["name"].replace(" [Loopback]", ""), int(d["defaultSampleRate"]),
                          int(d["maxInputChannels"])) for d in p.get_loopback_device_info_generator()]
        devices.sort(key=lambda d: d.index != default["index"])
        return devices
    finally:
        p.terminate()


def list_audio_apps() -> list[AudioApp]:
    """目前有音訊工作階段（正在或剛發出聲音）的程式。同一個程式只列一次。"""
    from pycaw.pycaw import AudioUtilities

    seen: dict[str, AudioApp] = {}
    for session in AudioUtilities.GetAllSessions():
        proc = session.Process
        if proc is None:
            continue
        try:
            name = proc.name()
        except Exception:  # noqa: BLE001 — 程式剛好結束
            continue
        if name.lower() not in ("svchost.exe", "audiodg.exe"):
            seen.setdefault(name.lower(), AudioApp(proc.pid, name))
    return sorted(seen.values(), key=lambda a: a.name.lower())


class _Downmixer:
    """多聲道、任意取樣率 → 16 kHz 單聲道（串流重新取樣，塊與塊之間不會有接縫）。"""

    def __init__(self, rate: int, channels: int):
        import soxr

        self.channels = channels
        self.resampler = None if rate == SAMPLE_RATE else soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32")

    def __call__(self, raw: bytes) -> np.ndarray:
        audio = np.frombuffer(raw, np.float32)
        if self.channels > 1:
            audio = audio.reshape(-1, self.channels).mean(axis=1)
        if self.resampler is not None:
            audio = self.resampler.resample_chunk(audio)
        return audio.astype(np.float32, copy=False)


class _PortAudioSource:
    """麥克風或系統聲音（PyAudioWPatch，WASAPI）。"""

    def __init__(self, device: Device):
        self.device = device
        self._pa = None
        self._stream = None

    def start(self, on_audio: AudioFn) -> None:
        import pyaudiowpatch as pa

        self._pa = pa.PyAudio()
        mix = _Downmixer(self.device.rate, self.device.channels)

        def callback(data, frames, time_info, status):
            audio = mix(data)
            if len(audio):
                on_audio(audio)
            return None, pa.paContinue

        self._stream = self._pa.open(format=pa.paFloat32, channels=self.device.channels, rate=self.device.rate,
                                     input=True, input_device_index=self.device.index,
                                     frames_per_buffer=self.device.rate // 10, stream_callback=callback)
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None


class MicSource(_PortAudioSource):
    pass


class LoopbackSource(_PortAudioSource):
    pass


class ProcessSource:
    def __init__(self, app: AudioApp):
        from omni2.audio.process_loopback import ProcessLoopbackCapture

        self.app = app
        self._capture = ProcessLoopbackCapture(app.pid)

    def start(self, on_audio: AudioFn) -> None:
        self._capture.start(on_audio)

    def stop(self) -> None:
        self._capture.stop()


class FileSource:
    """把音檔以真實速度一塊一塊送出，像有人在播放一樣。測試和重現問題用。"""

    def __init__(self, path: str, chunk_ms: int = 100, speed: float = 1.0):
        from omni2.audio.decode import read_wav

        self.audio = read_wav(path)
        self.chunk = SAMPLE_RATE * chunk_ms // 1000
        self.interval = chunk_ms / 1000 / speed
        self._stop = None
        self._thread = None

    def start(self, on_audio: AudioFn) -> None:
        import threading
        import time

        self._stop = threading.Event()

        def run() -> None:
            next_t = time.monotonic()
            for i in range(0, len(self.audio), self.chunk):
                if self._stop.is_set():
                    return
                on_audio(self.audio[i:i + self.chunk])
                next_t += self.interval
                time.sleep(max(0.0, next_t - time.monotonic()))

        self._thread = threading.Thread(target=run, name="file-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=2)
