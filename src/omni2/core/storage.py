"""資料資料夾的空間使用與搬移。

搬移的東西：工作檔（jobs）、模型（models）、llama.cpp（bin），以及（選擇的話）整個 Hugging Face
快取。同一顆磁碟直接改名；跨磁碟逐檔複製並回報進度，全部複製完才刪除原本的檔案。
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from omni2.config.settings import Settings, apply_storage, save_settings
from omni2.core import paths

DATA_PARTS = (("jobs", "工作檔（中間音訊、斷點）"), ("models", "翻譯與其他模型"), ("bin", "llama.cpp 程式"))
ProgressFn = Callable[[int, int, str], None]  # （已完成位元組, 總位元組, 目前的檔案）


class StorageCancelled(Exception):
    pass


@dataclass
class UsageItem:
    label: str
    path: Path
    size: int


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def default_hf_home() -> Path:
    """系統預設的 Hugging Face 快取位置（不受我們的設定影響）。"""
    return Path(os.environ.get("OMNI2_DEFAULT_HF_HOME") or Path.home() / ".cache" / "huggingface")


def current_hf_home(settings: Settings) -> Path:
    return paths.hf_dir() if settings.storage.hf_in_data_dir else default_hf_home()


def usage(settings: Settings) -> list[UsageItem]:
    root = paths.data_dir()
    items = [UsageItem(label, root / name, dir_size(root / name)) for name, label in DATA_PARTS]
    hf = current_hf_home(settings)
    items.append(UsageItem("語音辨識模型（Hugging Face 快取）", hf, dir_size(hf)))
    return items


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def _same_drive(a: Path, b: Path) -> bool:
    return a.resolve().drive.casefold() == b.resolve().drive.casefold()


def move_tree(src: Path, dst: Path, progress: ProgressFn | None = None, done: int = 0, total: int = 0,
              cancel: threading.Event | None = None) -> int:
    """把 src 資料夾的內容搬到 dst（dst 已存在時合併，同名檔案以 src 為準）。回傳累計位元組。"""
    if not src.exists():
        return done
    dst.mkdir(parents=True, exist_ok=True)
    same_drive = _same_drive(src, dst)
    if same_drive and not any(dst.iterdir()):
        size = dir_size(src)
        dst.rmdir()
        os.replace(src, dst)  # 同一顆磁碟：整個資料夾直接改名
        if progress:
            progress(done + size, total, str(dst))
        return done + size
    for root, _, files in os.walk(src):
        target_root = dst / Path(root).relative_to(src)
        target_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            if cancel is not None and cancel.is_set():
                raise StorageCancelled()
            source = Path(root) / name
            size = source.stat().st_size
            if same_drive:
                os.replace(source, target_root / name)
            else:
                shutil.copy2(source, target_root / name)
            done += size
            if progress:
                progress(done, total, name)
    shutil.rmtree(src)  # 全部搬完才刪除原本的資料夾
    return done


def _rebase(value: str, old: Path, new: Path) -> str:
    if not value:
        return value
    try:
        rel = Path(value).resolve().relative_to(old.resolve())
    except ValueError:
        return value  # 不在舊資料夾裡（例如放在別處的 GGUF），不動
    return str(new / rel)


def relocate(settings: Settings, new_dir: Path, hf_in_data_dir: bool,
             progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> None:
    """把資料搬到 new_dir 並更新設定。new_dir 等於預設位置時，設定改回空字串。"""
    old_dir = paths.data_dir()
    new_dir = new_dir.resolve()
    old_hf = current_hf_home(settings)
    new_hf = (new_dir / "huggingface") if hf_in_data_dir else default_hf_home()

    moves = [(old_dir / name, new_dir / name) for name, _ in DATA_PARTS if old_dir.resolve() != new_dir]
    if old_hf.resolve() != new_hf.resolve():
        moves.append((old_hf, new_hf))
    for src, dst in moves:  # 動手前先全部檢查，避免搬到一半才發現不行
        if dst.resolve().is_relative_to(src.resolve()):
            raise ValueError(f"不能搬到原本資料夾的裡面：{dst}")
    total = sum(dir_size(src) for src, _ in moves)
    done = 0
    for src, dst in moves:
        done = move_tree(src, dst, progress, done, total, cancel)

    t = settings.translate
    t.model_path = _rebase(t.model_path, old_dir, new_dir)
    t.server_exe = _rebase(t.server_exe, old_dir, new_dir)
    settings.terms.model_path = _rebase(settings.terms.model_path, old_dir, new_dir)
    default = paths.app_dir().resolve()
    settings.storage.data_dir = "" if new_dir == default else str(new_dir)
    settings.storage.hf_in_data_dir = hf_in_data_dir
    apply_storage(settings)
    save_settings(settings)


def clear_jobs() -> int:
    """刪除所有工作檔（中間音訊、斷點）。回傳釋放的位元組。"""
    root = paths.jobs_dir()
    size = dir_size(root)
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    return size
