"""意見回報：開啟 GitHub 的回報表單，並自動帶入版本和硬體資訊。

只帶入辨識問題需要的資訊（版本、Windows、顯示卡、使用的引擎與模型名稱），不帶任何檔案路徑或金鑰。
表單欄位的 id 要和 .github/ISSUE_TEMPLATE 裡的一致。
"""

from __future__ import annotations

import platform
from pathlib import Path
from urllib.parse import urlencode

from omni2 import __version__
from omni2.config.settings import Settings
from omni2.core.hardware import Hardware

REPO_URL = "https://github.com/Oscar0711/mimiyaku"
DISCUSSIONS_URL = f"{REPO_URL}/discussions"
TEMPLATES = {"bug": "bug_report.yml", "feature": "feature_request.yml"}


def environment(settings: Settings, hw: Hardware | None) -> str:
    s = settings
    lines = [f"Mimiyaku {__version__}", f"Windows：{platform.version()}（{platform.machine()}）"]
    if hw is not None:
        gpu = hw.gpu
        lines.append(f"顯示卡：{gpu.name}（{gpu.vram_gb:.0f} GB），驅動 {hw.driver}，CUDA {hw.driver_cuda:g}"
                     if gpu else "顯示卡：沒有 NVIDIA 顯示卡")
        if hw.ram_gb:
            lines.append(f"記憶體：{hw.ram_gb:.0f} GB")
    translate = "外部服務" if s.translate.endpoint else (Path(s.translate.model_path).name or "未設定")
    lines.append(f"檔案翻譯：{s.stt.engine}（{s.stt.active_model()}）＋{translate}")
    live_translate = Path(s.live.translate_model_path).name if s.live.translate_model_path else "同上"
    lines.append(f"即時翻譯：{s.live.stt_engine}＋{live_translate}")
    return "\n".join(lines)


def issue_url(kind: str, settings: Settings, hw: Hardware | None) -> str:
    """kind：bug（回報問題）或 feature（功能建議）。"""
    params = {"template": TEMPLATES[kind]}
    if kind == "bug":
        params["environment"] = environment(settings, hw)
    return f"{REPO_URL}/issues/new?{urlencode(params)}"
