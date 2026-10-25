"""只擷取指定程式（含它的子程式）發出的聲音：Windows 的 process loopback（Windows 10 20348+／Windows 11）。

用 comtypes 直接呼叫 WASAPI，不需要另外編譯原生程式：
ActivateAudioInterfaceAsync("VAD\\Process_Loopback", 目標 PID) → IAudioClient（要求系統直接轉成
16 kHz 單聲道 16-bit）→ 事件驅動讀取 IAudioCaptureClient。
"""

from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import POINTER, Structure, byref, c_int, c_int64, c_uint32, c_uint64, c_ushort, c_void_p, wintypes
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

_WAVE_FORMAT_PCM = 1
_AUDCLNT_SHAREMODE_SHARED = 0
_AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
_AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
_AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = 0x80000000
_AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY = 0x08000000
_AUDCLNT_BUFFERFLAGS_SILENT = 0x2
_AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
_PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0
_PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE = 1
_VT_BLOB = 65
_VIRTUAL_DEVICE = "VAD\\Process_Loopback"


class _WaveFormatEx(Structure):
    _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD), ("nSamplesPerSec", wintypes.DWORD),
                ("nAvgBytesPerSec", wintypes.DWORD), ("nBlockAlign", wintypes.WORD),
                ("wBitsPerSample", wintypes.WORD), ("cbSize", wintypes.WORD)]


class _LoopbackParams(Structure):
    _fields_ = [("TargetProcessId", wintypes.DWORD), ("ProcessLoopbackMode", c_int)]


class _ActivationParams(Structure):
    _fields_ = [("ActivationType", c_int), ("ProcessLoopbackParams", _LoopbackParams)]


class _Blob(Structure):
    _fields_ = [("cbSize", wintypes.ULONG), ("pBlobData", c_void_p)]


class _PropVariant(Structure):
    # PROPVARIANT 只用到 VT_BLOB：vt 和三個保留欄位之後就是 BLOB
    _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort), ("r3", c_ushort), ("blob", _Blob)]


def _interfaces():
    """延遲定義 COM 介面：import comtypes 會初始化 COM，只在真的要擷取時才做。"""
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

    class IAudioClient(IUnknown):
        _iid_ = GUID("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")
        _methods_ = [
            COMMETHOD([], HRESULT, "Initialize", (["in"], c_int, "ShareMode"), (["in"], wintypes.DWORD, "StreamFlags"),
                      (["in"], c_int64, "hnsBufferDuration"), (["in"], c_int64, "hnsPeriodicity"),
                      (["in"], POINTER(_WaveFormatEx), "pFormat"), (["in"], POINTER(GUID), "AudioSessionGuid")),
            COMMETHOD([], HRESULT, "GetBufferSize", (["out"], POINTER(c_uint32), "pNumBufferFrames")),
            COMMETHOD([], HRESULT, "GetStreamLatency", (["out"], POINTER(c_int64), "phnsLatency")),
            COMMETHOD([], HRESULT, "GetCurrentPadding", (["out"], POINTER(c_uint32), "pNumPaddingFrames")),
            COMMETHOD([], HRESULT, "IsFormatSupported", (["in"], c_int, "ShareMode"),
                      (["in"], POINTER(_WaveFormatEx), "pFormat"),
                      (["out"], POINTER(POINTER(_WaveFormatEx)), "ppClosestMatch")),
            COMMETHOD([], HRESULT, "GetMixFormat", (["out"], POINTER(POINTER(_WaveFormatEx)), "ppDeviceFormat")),
            COMMETHOD([], HRESULT, "GetDevicePeriod", (["out"], POINTER(c_int64), "phnsDefaultDevicePeriod"),
                      (["out"], POINTER(c_int64), "phnsMinimumDevicePeriod")),
            COMMETHOD([], HRESULT, "Start"),
            COMMETHOD([], HRESULT, "Stop"),
            COMMETHOD([], HRESULT, "Reset"),
            COMMETHOD([], HRESULT, "SetEventHandle", (["in"], wintypes.HANDLE, "eventHandle")),
            COMMETHOD([], HRESULT, "GetService", (["in"], POINTER(GUID), "riid"), (["out"], POINTER(c_void_p), "ppv")),
        ]

    class IAudioCaptureClient(IUnknown):
        _iid_ = GUID("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")
        _methods_ = [
            COMMETHOD([], HRESULT, "GetBuffer", (["out"], POINTER(c_void_p), "ppData"),
                      (["out"], POINTER(c_uint32), "pNumFramesToRead"), (["out"], POINTER(wintypes.DWORD), "pdwFlags"),
                      (["out"], POINTER(c_uint64), "pu64DevicePosition"),
                      (["out"], POINTER(c_uint64), "pu64QPCPosition")),
            COMMETHOD([], HRESULT, "ReleaseBuffer", (["in"], c_uint32, "NumFramesRead")),
            COMMETHOD([], HRESULT, "GetNextPacketSize", (["out"], POINTER(c_uint32), "pNumFramesInNextPacket")),
        ]

    class IActivateAudioInterfaceAsyncOperation(IUnknown):
        _iid_ = GUID("{72A22D78-CDE4-431D-B8CC-843A71199B6D}")
        _methods_ = [
            COMMETHOD([], HRESULT, "GetActivateResult", (["out"], POINTER(HRESULT), "activateResult"),
                      (["out"], POINTER(POINTER(IUnknown)), "activatedInterface")),
        ]

    class IActivateAudioInterfaceCompletionHandler(IUnknown):
        _iid_ = GUID("{41D949AB-9862-444A-80F6-C261334DA5EB}")
        _methods_ = [
            COMMETHOD([], HRESULT, "ActivateCompleted",
                      (["in"], POINTER(IActivateAudioInterfaceAsyncOperation), "activateOperation")),
        ]

    class IAgileObject(IUnknown):
        # 完成通知會在其他執行緒呼叫，處理器必須是 agile 物件
        _iid_ = GUID("{94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90}")
        _methods_ = []

    return (IAudioClient, IAudioCaptureClient, IActivateAudioInterfaceAsyncOperation,
            IActivateAudioInterfaceCompletionHandler, IAgileObject)


class ProcessLoopbackCapture:
    """擷取某個程式（預設包含它的子程式）的聲音，以 16 kHz 單聲道 float32 回呼。"""

    def __init__(self, pid: int, include_tree: bool = True, chunk_ms: int = 100):
        self.pid = pid
        self.include_tree = include_tree
        self.chunk = SAMPLE_RATE * chunk_ms // 1000
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._ready = threading.Event()

    def start(self, on_audio: Callable[[np.ndarray], None]) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(on_audio,), name="process-loopback", daemon=True)
        self._thread.start()
        self._ready.wait(10)
        if self._error is not None:
            raise RuntimeError(f"無法擷取程式（PID {self.pid}）的聲音：{self._error}") from self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def _activate(self):
        import comtypes
        from comtypes import COMObject

        (IAudioClient, IAudioCaptureClient, IAsyncOperation,
         ICompletionHandler, IAgileObject) = _interfaces()

        done = threading.Event()

        class Handler(COMObject):
            _com_interfaces_ = [ICompletionHandler, IAgileObject]

            def ActivateCompleted(self, operation):  # noqa: N802 — COM 方法名稱
                done.set()
                return 0

        params = _ActivationParams(
            _AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
            _LoopbackParams(self.pid, _PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE if self.include_tree
                            else _PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE),
        )
        prop = _PropVariant(vt=_VT_BLOB)
        prop.blob.cbSize = ctypes.sizeof(params)
        prop.blob.pBlobData = ctypes.cast(ctypes.pointer(params), c_void_p)

        activate = ctypes.WinDLL("Mmdevapi").ActivateAudioInterfaceAsync
        activate.argtypes = [wintypes.LPCWSTR, POINTER(comtypes.GUID), POINTER(_PropVariant),
                             POINTER(ICompletionHandler), POINTER(POINTER(IAsyncOperation))]
        activate.restype = comtypes.HRESULT
        handler = Handler()
        operation = POINTER(IAsyncOperation)()
        activate(_VIRTUAL_DEVICE, byref(IAudioClient._iid_), byref(prop),
                 handler.QueryInterface(ICompletionHandler), byref(operation))
        if not done.wait(5):
            raise TimeoutError("等不到 Windows 的音訊啟用回應")
        hr, unknown = operation.GetActivateResult()
        if hr != 0:
            raise OSError(f"音訊啟用失敗（HRESULT 0x{hr & 0xFFFFFFFF:08X}）")
        return unknown.QueryInterface(IAudioClient), IAudioCaptureClient

    def _run(self, on_audio: Callable[[np.ndarray], None]) -> None:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateEventW.restype = wintypes.HANDLE
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        event = None
        client = None
        try:
            import comtypes

            # import comtypes 時就會把這條執行緒初始化成 STA；那樣也能用（完成通知的處理器是
            # agile 物件，會在其他執行緒被呼叫），所以模式已經設定過就沿用
            try:
                comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
            except OSError:
                pass
            client, ICaptureClient = self._activate()
            fmt = _WaveFormatEx(_WAVE_FORMAT_PCM, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, 0)
            flags = (_AUDCLNT_STREAMFLAGS_LOOPBACK | _AUDCLNT_STREAMFLAGS_EVENTCALLBACK
                     | _AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | _AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY)
            client.Initialize(_AUDCLNT_SHAREMODE_SHARED, flags, 200_0000, 0, byref(fmt), None)  # 200 ms 緩衝
            event = k32.CreateEventW(None, False, False, None)
            client.SetEventHandle(event)
            capture = ctypes.cast(client.GetService(byref(ICaptureClient._iid_)), POINTER(ICaptureClient))
            client.Start()
        except BaseException as e:  # noqa: BLE001 — 交給 start() 轉成清楚的錯誤
            self._error = e
            self._ready.set()
            if event:
                k32.CloseHandle(event)
            return
        self._ready.set()
        log.info("開始擷取 PID %d 的聲音", self.pid)
        pending: list[np.ndarray] = []
        pending_len = 0
        try:
            while not self._stop.is_set():
                k32.WaitForSingleObject(event, 100)
                while capture.GetNextPacketSize() > 0:
                    data, frames, flags_, _, _ = capture.GetBuffer()
                    if flags_ & _AUDCLNT_BUFFERFLAGS_SILENT or not data:
                        block = np.zeros(frames, np.float32)
                    else:
                        raw = ctypes.string_at(data, frames * 2)
                        block = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                    capture.ReleaseBuffer(frames)
                    pending.append(block)
                    pending_len += frames
                    if pending_len >= self.chunk:
                        on_audio(np.concatenate(pending))
                        pending, pending_len = [], 0
        except BaseException as e:  # noqa: BLE001 — 例如目標程式結束
            log.warning("擷取 PID %d 的聲音時中斷：%s", self.pid, e)
        finally:
            try:
                client.Stop()
            except Exception:  # noqa: BLE001
                pass
            k32.CloseHandle(event)
