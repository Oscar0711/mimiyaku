"""設定。預設值寫在程式裡，使用者的修改存到 %LOCALAPPDATA%\\OmniTranslate2\\config.yaml。"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from omni2.core import paths
from omni2.core.jsonio import write_text_atomic


class SttSettings(BaseModel):
    engine: str = "qwen3-asr"  # qwen3-asr | faster-whisper（M2 實測 Qwen3-ASR 錯誤率較低、標點完整）
    model: str = "large-v3-turbo"  # faster-whisper 的模型
    qwen_model: str = "Qwen/Qwen3-ASR-1.7B"
    qwen_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None  # None＝自動偵測
    vad_preset: str = "file"
    beam_size: int = 5
    batch_size: int = 8  # faster-whisper 批次推論；1＝不用批次
    filter_hallucinations: bool = True

    def active_model(self) -> str:
        return self.qwen_model if self.engine == "qwen3-asr" else self.model


class TranslateSettings(BaseModel):
    enabled: bool = True
    target: str = "zh-TW"
    # 填了就改用外部的 OpenAI 相容服務（Ollama、LM Studio…），不啟動 llama-server
    endpoint: str | None = None
    endpoint_model: str = "local"
    server_exe: str = ""
    model_path: str = ""
    ctx_size: int = 8192
    gpu_layers: int = 99
    server_args: list[str] = Field(default_factory=list)
    # auto＝依模型檔名選擇：hy-mt、translategemma，其餘用 generic
    prompt_style: str = "auto"
    batch_lines: int = 10
    context_lines: int = 3
    temperature: float = 0.2
    max_retries: int = 2
    opencc: str = "s2twp"  # 目標是繁中時的繁體保險；空字串＝關閉


class LayoutSettings(BaseModel):
    # 寬度以半形字為單位，全形字算 2
    max_line_width: int = 42
    max_lines: int = 2
    max_duration: float = 7.0
    min_duration: float = 1.0
    max_cps: float = 9.0  # 每秒全形字數（半形字算半個）
    gap: float = 0.08


class SeparationSettings(BaseModel):
    # audio-separator 的模型檔名；人聲 SDR 最高的 Mel-Band RoFormer
    model: str = "vocals_mel_band_roformer.ckpt"


class DiarizationSettings(BaseModel):
    engine: str = "ecapa"  # ecapa（不需帳號）| pyannote（較準，需 Hugging Face 登入）
    model: str = "speechbrain/spkrec-ecapa-voxceleb"
    # ecapa 聚類的餘弦距離門檻：越小越容易把同一人拆成兩位講者，越大越容易把不同人併成一人
    threshold: float = 0.7


class StorageSettings(BaseModel):
    # 工作檔、模型、llama.cpp 的資料夾；空字串＝%LOCALAPPDATA%\OmniTranslate2
    data_dir: str = ""
    # 語音辨識模型（Hugging Face 快取）也放在資料夾裡；否則用系統預設的 ~/.cache/huggingface
    hf_in_data_dir: bool = False


class TermsSettings(BaseModel):
    # 抽術語用的模型；空字串＝和翻譯用同一個
    model_path: str = ""
    chunk_chars: int = 3000
    ctx_size: int = 8192
    review: bool = True  # CLI 在終端機裡時，抽完先讓使用者確認
    builtin: list[str] = Field(default_factory=lambda: ["streaming"])  # 內建術語表


class FileModeSettings(BaseModel):
    """檔案工作區上次使用的選項，下次開啟時沿用。"""

    src_lang: str | None = None
    translate: bool = True
    separate_vocals: bool = False
    diarize: bool = False
    num_speakers: int = 0  # 0＝自動判斷
    terms_first: bool = False


class OutputSettings(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["srt"])
    bilingual: bool = False
    dir: str | None = None  # None＝和原檔放在同一個資料夾
    timestamps: bool = True  # txt、md 逐字稿是否加時間戳


class Settings(BaseModel):
    stt: SttSettings = Field(default_factory=SttSettings)
    translate: TranslateSettings = Field(default_factory=TranslateSettings)
    layout: LayoutSettings = Field(default_factory=LayoutSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    separation: SeparationSettings = Field(default_factory=SeparationSettings)
    diarization: DiarizationSettings = Field(default_factory=DiarizationSettings)
    terms: TermsSettings = Field(default_factory=TermsSettings)
    file_mode: FileModeSettings = Field(default_factory=FileModeSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)


def apply_storage(settings: Settings) -> None:
    """讓這個行程和之後 spawn 的子行程都使用設定的資料資料夾。"""
    import os

    paths.set_data_dir(settings.storage.data_dir or None)
    # 記下使用者原本的 Hugging Face 快取位置，切回「不放在資料夾」時才能還原
    if "OMNI2_DEFAULT_HF_HOME" not in os.environ:
        os.environ["OMNI2_DEFAULT_HF_HOME"] = os.environ.get("HF_HOME") or str(
            Path.home() / ".cache" / "huggingface")
    if settings.storage.hf_in_data_dir:
        os.environ["HF_HOME"] = str(paths.hf_dir())
    else:
        os.environ["HF_HOME"] = os.environ["OMNI2_DEFAULT_HF_HOME"]


def find_llama_server() -> str:
    """找 bin\\llama.cpp 底下最新一版的 llama-server.exe。"""
    root = paths.bin_dir() / "llama.cpp"
    found = sorted(root.glob("*/llama-server.exe"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(found[0]) if found else ""


def load_settings(path: Path | None = None) -> Settings:
    path = path or paths.config_path()
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        settings = Settings.model_validate(data)
    else:
        settings = Settings()
    apply_storage(settings)
    if not settings.translate.server_exe or not Path(settings.translate.server_exe).exists():
        detected = find_llama_server()
        if detected:
            settings.translate.server_exe = detected
    if not path.exists():
        save_settings(settings, path)
    return settings


def save_settings(settings: Settings, path: Path | None = None) -> None:
    path = path or paths.config_path()
    text = yaml.safe_dump(settings.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
    write_text_atomic(path, text)
