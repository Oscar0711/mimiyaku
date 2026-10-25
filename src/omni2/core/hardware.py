"""偵測硬體：NVIDIA 顯示卡（型號、顯示記憶體、驅動支援的 CUDA 版本）、CPU、記憶體、磁碟空間。

只用 nvidia-smi 和 Windows API，不 import torch，主視窗開啟時也能很快查完。
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Gpu:
    name: str
    vram_mb: int
    compute_cap: float = 0.0  # 例如 12.0（Blackwell）、8.9（Ada）

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024


@dataclass
class Hardware:
    cpu: str = ""
    ram_gb: float = 0.0
    gpus: list[Gpu] = field(default_factory=list)
    driver: str = ""
    driver_cuda: float = 0.0  # 驅動程式支援的最高 CUDA 版本；0＝沒有 NVIDIA 驅動
    disk_free_gb: float = 0.0  # 資料資料夾所在磁碟的剩餘空間

    @property
    def gpu(self) -> Gpu | None:
        """顯示記憶體最大的那張。"""
        return max(self.gpus, key=lambda g: g.vram_mb, default=None)

    def describe(self) -> list[str]:
        lines = []
        if self.gpu:
            g = self.gpu
            lines.append(f"顯示卡：{g.name}（{g.vram_gb:.0f} GB 顯示記憶體）")
            lines.append(f"驅動程式：{self.driver or '未知'}，支援 CUDA {self.driver_cuda:g}")
        else:
            lines.append("顯示卡：沒有偵測到 NVIDIA 顯示卡")
        if self.cpu:
            lines.append(f"CPU：{self.cpu}")
        if self.ram_gb:
            lines.append(f"記憶體：{self.ram_gb:.0f} GB")
        if self.disk_free_gb:
            lines.append(f"資料資料夾所在磁碟剩餘：{self.disk_free_gb:.0f} GB")
        return lines


def parse_gpus(csv: str) -> list[Gpu]:
    """解析 nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits。"""
    gpus = []
    for line in csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            vram = int(float(parts[1]))
        except ValueError:
            continue
        try:
            cc = float(parts[2]) if len(parts) > 2 else 0.0
        except ValueError:
            cc = 0.0
        gpus.append(Gpu(parts[0], vram, cc))
    return gpus


def parse_driver_cuda(banner: str) -> float:
    """從 nvidia-smi 的表頭找「CUDA Version: 12.4」（新版驅動寫成「CUDA UMD Version: 13.4」）。"""
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", banner)
    return float(m.group(1)) if m else 0.0


def _run(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW,
                              encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _ram_gb() -> float:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / 2**30
    except (AttributeError, OSError):
        pass
    return 0.0


def _cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        return ""


def detect(data_dir: Path | None = None) -> Hardware:
    hw = Hardware(cpu=_cpu_name(), ram_gb=_ram_gb())
    smi = shutil.which("nvidia-smi")
    if smi:
        hw.gpus = parse_gpus(_run([smi, "--query-gpu=name,memory.total,compute_cap",
                                   "--format=csv,noheader,nounits"]))
        drivers = _run([smi, "--query-gpu=driver_version", "--format=csv,noheader"]).split()
        hw.driver = drivers[0] if drivers else ""
        hw.driver_cuda = parse_driver_cuda(_run([smi]))
    if data_dir is not None:
        try:
            hw.disk_free_gb = shutil.disk_usage(data_dir).free / 2**30
        except OSError:
            pass
    return hw
