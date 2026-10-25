"""在桌面或開始功能表建立啟動捷徑（指向虛擬環境裡的 mimiyaku-gui.exe，不開主控台視窗）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

NAME = "Mimiyaku.lnk"
LEGACY_NAMES = ("OmniTranslate.lnk",)  # 改名前建立的捷徑，建新的時順便刪掉
APP_ID = "Mimiyaku"  # 工作列用來辨認視窗屬於哪個程式，捷徑和視窗要一致


def icon_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "omni2.ico"


def launch_command() -> tuple[str, str]:
    """（執行檔, 參數）。uv 安裝的 gui-scripts 會在 Scripts 底下產生 mimiyaku-gui.exe。"""
    scripts = Path(sys.executable).parent
    for name in ("mimiyaku-gui.exe", "omni2-gui.exe"):  # 改名前的環境只有 omni2-gui.exe
        gui = scripts / name
        if gui.exists():
            return str(gui), ""
    pythonw = scripts / "pythonw.exe"
    return str(pythonw if pythonw.exists() else sys.executable), "-m omni2.ui.app"


def desktop_dir() -> Path:
    from platformdirs import user_desktop_dir

    return Path(user_desktop_dir())


def start_menu_dir() -> Path:
    return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Microsoft" / "Windows" \
        / "Start Menu" / "Programs"


def shortcut_path(folder: Path) -> Path:
    return folder / NAME


def create(folder: Path) -> Path:
    """建立（或覆寫）捷徑，回傳捷徑的路徑。"""
    import comtypes.client

    exe, args = launch_command()
    folder.mkdir(parents=True, exist_ok=True)
    path = shortcut_path(folder)
    shell = comtypes.client.CreateObject("WScript.Shell", dynamic=True)
    link = shell.CreateShortcut(str(path))
    link.TargetPath = exe
    link.Arguments = args
    link.WorkingDirectory = str(Path(sys.prefix).parent)  # 專案資料夾（.venv 的上一層）
    link.Description = "Mimiyaku：語音辨識與字幕翻譯"
    if icon_path().exists():
        link.IconLocation = f"{icon_path()},0"
    link.Save()
    for old in LEGACY_NAMES:
        (folder / old).unlink(missing_ok=True)
    return path


def set_app_id() -> None:
    """讓工作列顯示我們的圖示，而不是 Python 的。"""
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except (AttributeError, OSError):
        pass
