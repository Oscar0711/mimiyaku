"""檔案位置。

* 設定檔、術語表、記錄檔：固定在 %LOCALAPPDATA%\\Mimiyaku（程式要先讀到設定才知道其他東西在哪）。
  改名前的版本用 %LOCALAPPDATA%\\OmniTranslate2；那個資料夾已經存在時繼續沿用，模型和設定不用搬。
* 工作檔（jobs）、模型（models）、llama.cpp（bin）：放在「資料資料夾」，預設同上，可以在設定裡改。

資料資料夾透過環境變數 OMNI2_DATA_DIR 傳遞：載入設定時設定好，spawn 出來的子行程會繼承。
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "Mimiyaku"
LEGACY_APP_NAME = "OmniTranslate2"  # 改名前的資料夾
DATA_ENV = "OMNI2_DATA_DIR"


def app_dir() -> Path:
    path = Path(user_data_dir(APP_NAME, appauthor=False, roaming=False))
    legacy = Path(user_data_dir(LEGACY_APP_NAME, appauthor=False, roaming=False))
    if legacy.exists() and not path.exists():
        return legacy
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    custom = os.environ.get(DATA_ENV)
    path = Path(custom) if custom else app_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_data_dir(path: str | os.PathLike | None) -> None:
    if path:
        os.environ[DATA_ENV] = str(path)
    else:
        os.environ.pop(DATA_ENV, None)


def _sub(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def jobs_dir() -> Path:
    return _sub(data_dir(), "jobs")


def models_dir() -> Path:
    return _sub(data_dir(), "models")


def bin_dir() -> Path:
    return _sub(data_dir(), "bin")


def hf_dir() -> Path:
    """「語音辨識模型也放在資料資料夾」時的 Hugging Face 快取位置（HF_HOME）。"""
    return data_dir() / "huggingface"


def url_output_dir() -> Path:
    """網址輸入沒有「原檔所在的資料夾」，字幕預設存到「影片」資料夾下的 Mimiyaku。"""
    from platformdirs import user_videos_dir

    return Path(user_videos_dir()) / "Mimiyaku"


def logs_dir() -> Path:
    return _sub(app_dir(), "logs")


def config_path() -> Path:
    return app_dir() / "config.yaml"


def glossary_path() -> Path:
    return app_dir() / "glossary.yaml"
