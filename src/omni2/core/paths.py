"""程式資料的存放位置：%LOCALAPPDATA%\\OmniTranslate2。"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "OmniTranslate2"


def app_dir() -> Path:
    path = Path(user_data_dir(APP_NAME, appauthor=False, roaming=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sub(name: str) -> Path:
    path = app_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def jobs_dir() -> Path:
    return _sub("jobs")


def logs_dir() -> Path:
    return _sub("logs")


def bin_dir() -> Path:
    return _sub("bin")


def config_path() -> Path:
    return app_dir() / "config.yaml"


def glossary_path() -> Path:
    return app_dir() / "glossary.yaml"
