"""引擎選單下方的提示：說明本地辨識實際會用哪個模型（韓文會換成韓文微調版，見 omni2.stt.language_models）。

檔案翻譯和即時翻譯共用。
"""

from __future__ import annotations

from omni2.core import lang
from omni2.models import catalog
from omni2.stt.language_models import korean_model


def model_label(repo: str) -> str:
    m = catalog.find_repo(repo)
    return m.label if m else repo.rsplit("/", 1)[-1]


def local_stt_hint(engine: str, source: str | None, qwen_model: str, ko_model: str, whisper_model: str,
                   extra_vram: str = "") -> str:
    """本地引擎的提示；雲端引擎回傳空字串。extra_vram：換成韓文微調版時多用的顯示記憶體說明。"""
    if engine == "faster-whisper":
        return f"使用 Whisper {whisper_model}。"
    if engine != "qwen3-asr":
        return ""
    name = model_label(qwen_model)
    korean = lang.base(source) == "ko"
    if korean and korean_model(source, ko_model):
        note = f"{extra_vram}，" if extra_vram else ""
        return f"韓文會改用「{model_label(ko_model)}」（{note}可以在「設定 → 辨識與翻譯」關掉）。"
    if korean and ko_model:
        return f"使用 {name}。下載「{model_label(ko_model)}」後，韓文會自動改用它，比較準（設定 → 硬體與模型）。"
    if korean:
        return f"使用 {name}（韓文微調版已在「設定 → 辨識與翻譯」關掉）。"
    if not ko_model:
        return f"使用 {name}。"
    if source is None:
        return f"使用 {name}；自動偵測時不會換成韓文微調版，韓文影片請把來源語言選韓文。"
    return f"使用 {name}；來源語言選韓文時會改用韓文微調版。"
