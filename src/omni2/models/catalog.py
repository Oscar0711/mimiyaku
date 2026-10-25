"""可以下載的模型：來源、大小、放在哪裡、怎麼判斷已經裝好。

四種來源：
* hf_repo：整個 Hugging Face repo，放在 Hugging Face 快取（HF_HOME），引擎用 repo 名稱載入
* hf_file：Hugging Face repo 裡的單一檔案（GGUF），放在資料資料夾的 models
* separator：audio-separator 的人聲分離模型，放在 models\\separator
* llama：llama.cpp 的 llama-server（GitHub 發行版），放在 bin\\llama.cpp\\<版本>
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from omni2.core import paths

GB = 2**30
MB = 2**20


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    group: str  # stt | translate | other
    kind: str  # hf_repo | hf_file | separator | llama
    size: int  # 約略大小（位元組），下載前顯示用
    repo: str = ""
    filename: str = ""
    note: str = ""
    urls: tuple[str, ...] = ()  # separator：依序嘗試的網址前綴


GROUPS = {"stt": "語音辨識", "translate": "翻譯", "other": "其他"}

SEPARATOR_URLS = (
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models",
    "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs",
)

CATALOG: list[ModelInfo] = [
    ModelInfo("qwen3-asr-1.7b", "Qwen3-ASR 1.7B", "stt", "hf_repo", int(4.38 * GB), "Qwen/Qwen3-ASR-1.7B",
              note="檔案翻譯預設，最準、標點完整"),
    ModelInfo("qwen3-asr-0.6b", "Qwen3-ASR 0.6B", "stt", "hf_repo", int(1.75 * GB), "Qwen/Qwen3-ASR-0.6B",
              note="即時翻譯預設，約 2 GB 顯示記憶體"),
    ModelInfo("qwen3-asr-1.7b-ko", "Qwen3-ASR 1.7B 韓文微調版", "stt", "hf_repo", int(4.39 * GB),
              "TeamUNIVA/qwen3_asr_1.7b_ko", note="下載後，來源語言是韓文時自動使用（實測錯字少約 12%）"),
    ModelInfo("qwen3-aligner", "Qwen3 ForcedAligner 0.6B", "stt", "hf_repo", int(1.71 * GB),
              "Qwen/Qwen3-ForcedAligner-0.6B", note="Qwen3-ASR 檔案模式的逐詞時間"),
    ModelInfo("whisper-turbo", "Whisper large-v3-turbo", "stt", "hf_repo", int(1.51 * GB),
              "mobiuslabsgmbh/faster-whisper-large-v3-turbo", note="faster-whisper 引擎"),
    ModelInfo("whisper-small", "Whisper small", "stt", "hf_repo", int(0.45 * GB), "Systran/faster-whisper-small",
              note="沒有 NVIDIA 顯示卡時用（CPU）"),
    ModelInfo("hy-mt2-7b-q6", "Hy-MT2 7B Q6_K", "translate", "hf_file", int(5.74 * GB), "tencent/Hy-MT2-7B-GGUF",
              "HY-MT2-7B-Q6_K.gguf", note="最準，約 7.1 GB 顯示記憶體"),
    ModelInfo("hy-mt2-7b-q4", "Hy-MT2 7B Q4_K_M", "translate", "hf_file", int(4.31 * GB), "tencent/Hy-MT2-7B-GGUF",
              "Hy-MT2-7B-Q4_K_M.gguf", note="約 5.7 GB 顯示記憶體（估計）"),
    ModelInfo("hy-mt2-1.8b-q8", "Hy-MT2 1.8B Q8_0", "translate", "hf_file", int(1.78 * GB),
              "tencent/Hy-MT2-1.8B-GGUF", "Hy-MT2-1.8B-Q8_0.gguf", note="小顯示卡用，約 2.6 GB（估計）"),
    ModelInfo("hy-mt2-1.8b-q4", "Hy-MT2 1.8B Q4_K_M", "translate", "hf_file", int(1.06 * GB),
              "tencent/Hy-MT2-1.8B-GGUF", "Hy-MT2-1.8B-Q4_K_M.gguf", note="約 1.8 GB；平均 chrF 40.7（7B Q6_K 是 48.3），快 3.7 倍"),
    ModelInfo("llama-server", "llama.cpp（llama-server）", "other", "llama", 557 * MB,
              note="翻譯模型的推論程式"),
    ModelInfo("ecapa", "ECAPA 講者特徵", "other", "hf_repo", 85 * MB, "speechbrain/spkrec-ecapa-voxceleb",
              note="講者分離（內建引擎）"),
    ModelInfo("separator", "Mel-Band RoFormer 人聲分離", "other", "separator", 871 * MB,
              filename="vocals_mel_band_roformer.ckpt", note="人聲分離；來源是 GitHub，可能比較慢",
              urls=SEPARATOR_URLS),
]

BY_ID = {m.id: m for m in CATALOG}


def get(model_id: str) -> ModelInfo:
    return BY_ID[model_id]


def find_file(filename: str) -> ModelInfo | None:
    """依 GGUF 檔名找目錄裡的模型（大小寫不分）。"""
    name = Path(filename).name.casefold()
    return next((m for m in CATALOG if m.kind == "hf_file" and m.filename.casefold() == name), None)


def find_repo(repo: str) -> ModelInfo | None:
    return next((m for m in CATALOG if m.kind == "hf_repo" and m.repo == repo), None)


# ---- 位置 ----

def hub_cache() -> Path:
    """Hugging Face 快取的 hub 資料夾（跟著 apply_storage 設定的 HF_HOME 走）。"""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def repo_dir(repo: str) -> Path:
    return hub_cache() / f"models--{repo.replace('/', '--')}"


def file_path(m: ModelInfo) -> Path:
    """hf_file 和 separator 下載後的位置。"""
    if m.kind == "separator":
        return paths.models_dir() / "separator" / m.filename
    return paths.models_dir() / m.filename


def llama_root() -> Path:
    return paths.bin_dir() / "llama.cpp"


def llama_versions() -> list[Path]:
    """已安裝的 llama.cpp 版本資料夾，新的在前。"""
    def build(p: Path) -> int:
        m = re.fullmatch(r"b(\d+)", p.name)
        return int(m.group(1)) if m else -1

    root = llama_root()
    if not root.exists():
        return []
    found = [p for p in root.iterdir() if (p / "llama-server.exe").exists()]
    return sorted(found, key=lambda p: (build(p), p.stat().st_mtime), reverse=True)


def _speechbrain_dir(m: ModelInfo) -> Path:
    return paths.models_dir() / "speechbrain" / m.repo.replace("/", "--")


# ---- 狀態 ----

def _regular_files(root: Path):
    for base, _, files in os.walk(root):
        for name in files:
            p = Path(base) / name
            if not p.is_symlink():
                yield p


def tree_size(root: Path) -> int:
    """資料夾裡實際的檔案大小（不算符號連結，Hugging Face 快取的 snapshots 可能全是連結）。"""
    total = 0
    for p in _regular_files(root):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _repo_complete(repo: str) -> bool:
    root = repo_dir(repo)
    if not root.exists() or any((root / "blobs").glob("*.incomplete")):
        return False
    for snap in (root / "snapshots").glob("*"):
        index = snap / "model.safetensors.index.json"
        if index.exists():  # 分成好幾個檔案的權重：每一份都要在
            try:
                shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            except (OSError, ValueError, KeyError):
                continue
            if all((snap / s).exists() for s in shards):
                return True
        elif any((snap / w).exists() for w in ("model.safetensors", "model.bin", "pytorch_model.bin",
                                                "embedding_model.ckpt")):
            return True
    return False


def repo_installed(repo: str) -> bool:
    """Hugging Face 快取裡有沒有完整的這個 repo（不在目錄裡的也可以查）。"""
    return _repo_complete(repo)


def is_installed(m: ModelInfo) -> bool:
    if m.kind == "hf_repo":
        if m.id == "ecapa" and (_speechbrain_dir(m) / "embedding_model.ckpt").exists():
            return True  # SpeechBrain 第一次載入時會複製一份到 models\speechbrain
        return _repo_complete(m.repo)
    if m.kind in ("hf_file", "separator"):
        return file_path(m).exists()
    return bool(llama_versions())


def installed_size(m: ModelInfo) -> int:
    if m.kind == "hf_repo":
        size = tree_size(repo_dir(m.repo))
        if m.id == "ecapa":
            size += tree_size(_speechbrain_dir(m))
        return size
    if m.kind == "hf_file":
        p = file_path(m)
        return p.stat().st_size if p.exists() else 0
    if m.kind == "separator":
        return sum(p.stat().st_size for p in file_path(m).parent.glob(Path(m.filename).stem + ".*"))
    return sum(tree_size(v) for v in llama_versions())


def installed_version(m: ModelInfo) -> str:
    if m.kind == "llama":
        versions = llama_versions()
        return versions[0].name if versions else ""
    return ""


def delete(m: ModelInfo) -> None:
    import shutil

    if m.kind == "hf_repo":
        shutil.rmtree(repo_dir(m.repo), ignore_errors=True)
        if m.id == "ecapa":
            shutil.rmtree(_speechbrain_dir(m), ignore_errors=True)
    elif m.kind == "hf_file":
        file_path(m).unlink(missing_ok=True)
    elif m.kind == "separator":
        for p in file_path(m).parent.glob(Path(m.filename).stem + ".*"):
            p.unlink(missing_ok=True)
    else:
        for v in llama_versions():
            shutil.rmtree(v, ignore_errors=True)


def delete_old_llama() -> int:
    """刪除最新一版以外的 llama.cpp。回傳釋放的位元組。"""
    import shutil

    freed = 0
    for old in llama_versions()[1:]:
        size = tree_size(old)
        shutil.rmtree(old, ignore_errors=True)
        if not old.exists():
            freed += size
    return freed
