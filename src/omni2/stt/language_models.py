"""依來源語言換用專門的辨識模型。

目前只有韓文：TeamUNIVA 的 Qwen3-ASR 1.7B 韓文微調版。用 3 部附人工字幕的韓文演講（35 分鐘）實測，
CER 從 4.03% 降到 3.53%，語速快的講者進步最多；作者說英文會稍微變差，所以只在韓文時使用。
只在模型已經下載時才換，不會在工作進行中自動下載 4 GB 的模型。來源語言是「自動偵測」時事先不知道
是韓文，不會換。
"""

from __future__ import annotations

from omni2.config.settings import SttSettings
from omni2.core import lang

KO_MODEL = "TeamUNIVA/qwen3_asr_1.7b_ko"


def korean_model(language: str | None, ko_model: str) -> str | None:
    """來源語言是韓文、設定了韓文模型、而且已經下載時，回傳要用的模型；否則 None。"""
    from omni2.models.catalog import repo_installed

    if not ko_model or lang.base(language) != "ko":
        return None
    return ko_model if repo_installed(ko_model) else None


def stt_for_language(stt: SttSettings, language: str | None) -> SttSettings:
    """檔案模式：Qwen3-ASR 遇到韓文時換成韓文微調版。"""
    if stt.engine != "qwen3-asr":
        return stt
    model = korean_model(language, stt.qwen_model_ko)
    return stt.model_copy(update={"qwen_model": model}) if model else stt
