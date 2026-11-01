"""資料夾瀏覽：列出資料夾裡的影音檔，勾選的檔案會加進佇列。"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omni2.export.formats import ALL_FORMATS

MEDIA_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac",
              ".mp4", ".mkv", ".webm", ".mov", ".avi", ".ts")
MAX_FILES = 2000  # 不小心開到整顆硬碟時別讓介面卡住

COL_NAME, COL_SIZE, COL_SUBS = range(3)
_PATH = Qt.ItemDataRole.UserRole


def list_media(folder: Path, recursive: bool) -> list[Path]:
    """資料夾裡的影音檔，依路徑排序；略過隱藏資料夾。"""
    found: list[Path] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".")) if recursive else []
        found.extend(Path(root) / f for f in files if Path(f).suffix.lower() in MEDIA_EXTS)
        if len(found) >= MAX_FILES:
            break
    return sorted(found[:MAX_FILES], key=lambda p: str(p).casefold())


def existing_outputs(media: Path, out_dir: Path) -> list[str]:
    """out_dir 裡已經有的字幕或逐字稿（檔名是「原檔名.語言.格式」）。"""
    prefix = media.stem.casefold() + "."
    try:
        names = os.listdir(out_dir)
    except OSError:
        return []
    return sorted(n for n in names
                  if n.casefold().startswith(prefix) and Path(n).suffix.lower().lstrip(".") in ALL_FORMATS)


def _size_text(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class FolderBrowser(QWidget):
    # 使用者勾選／取消勾選的檔案路徑
    toggled = Signal(list, bool)
    folder_changed = Signal(str, bool)

    def __init__(self, folder: str, recursive: bool, out_dir_for: Callable[[Path], Path], parent=None):
        super().__init__(parent)
        self.out_dir_for = out_dir_for
        self.folder: Path | None = None
        self._syncing = False  # 程式改勾選狀態時不要再發 toggled

        self.open_button = QPushButton("開啟資料夾…")
        self.open_button.clicked.connect(self._pick_folder)
        self.refresh_button = QPushButton("重新整理")
        self.refresh_button.clicked.connect(self.refresh)
        self.recursive_check = QCheckBox("包含子資料夾")
        self.recursive_check.setChecked(recursive)
        self.recursive_check.toggled.connect(self._on_recursive)
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        # 路徑很長時不要把整個面板撐寬（完整路徑在提示裡）
        self.path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bar = QHBoxLayout()
        bar.addWidget(self.open_button)
        bar.addWidget(self.refresh_button)
        bar.addWidget(self.recursive_check)
        bar.addStretch()

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(("資料夾裡的影音檔", "大小", "字幕"))
        self.tree.setRootIsDecorated(False)
        header = self.tree.header()
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_SIZE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_SUBS, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)
        self.tree.itemChanged.connect(self._on_item_changed)

        self.all_button = QPushButton("全選")
        self.all_button.clicked.connect(lambda: self._check_where(lambda item: True))
        self.new_button = QPushButton("只選沒字幕的")
        self.new_button.setToolTip("勾選輸出資料夾裡還沒有字幕的檔案")
        self.new_button.clicked.connect(lambda: self._check_where(lambda item: not item.text(COL_SUBS)))
        self.none_button = QPushButton("全不選")
        self.none_button.clicked.connect(lambda: self._set_all(False))
        self.count_label = QLabel()
        bottom = QHBoxLayout()
        for w in (self.all_button, self.new_button, self.none_button):
            bottom.addWidget(w)
        bottom.addStretch()
        bottom.addWidget(self.count_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.path_label)
        layout.addWidget(self.tree, 1)
        layout.addLayout(bottom)

        if folder and Path(folder).is_dir():
            self.set_folder(folder)
        else:
            self._show_empty()

    # ---- 資料夾 ----

    def _pick_folder(self) -> None:
        start = str(self.folder) if self.folder else ""
        folder = QFileDialog.getExistingDirectory(self, "要翻譯哪個資料夾裡的檔案", start)
        if folder:
            self.set_folder(folder)
            self.folder_changed.emit(str(self.folder), self.recursive_check.isChecked())

    def _on_recursive(self) -> None:
        self.refresh()
        if self.folder:
            self.folder_changed.emit(str(self.folder), self.recursive_check.isChecked())

    def set_folder(self, folder: str) -> None:
        self.folder = Path(folder)
        self.refresh()

    def _show_empty(self) -> None:
        self.path_label.setText("按「開啟資料夾」列出裡面可以翻譯的影音檔，勾選的檔案會加進右邊的佇列。")
        self.refresh_button.setEnabled(False)
        self._update_count()

    def refresh(self) -> None:
        """重新掃描資料夾；保留原本的勾選。"""
        if self.folder is None:
            return
        checked = set(self.checked_paths())
        self._syncing = True
        self.tree.clear()
        if not self.folder.is_dir():
            self._syncing = False
            self.path_label.setText(f"找不到資料夾：{self.folder}")
            self._update_count()
            return
        files = list_media(self.folder, self.recursive_check.isChecked())
        for path in files:
            try:
                size = _size_text(path.stat().st_size)
            except OSError:
                continue
            item = QTreeWidgetItem([str(path.relative_to(self.folder)), size, ""])
            item.setData(COL_NAME, _PATH, str(path))
            item.setToolTip(COL_NAME, str(path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(COL_NAME, Qt.CheckState.Checked if str(path) in checked
                               else Qt.CheckState.Unchecked)
            self.tree.addTopLevelItem(item)
        self._syncing = False
        self.refresh_outputs()
        more = f"（只列出前 {MAX_FILES} 個）" if len(files) >= MAX_FILES else ""
        self.path_label.setText(f"{self.folder}{more}")
        self.path_label.setToolTip(str(self.folder))
        self.refresh_button.setEnabled(True)
        self._update_count()

    def refresh_outputs(self) -> None:
        """更新「字幕」欄：輸出資料夾裡是否已經有這個檔案的字幕。"""
        for item in self._items():
            path = Path(item.data(COL_NAME, _PATH))
            outputs = existing_outputs(path, self.out_dir_for(path))
            item.setText(COL_SUBS, "已有" if outputs else "")
            item.setToolTip(COL_SUBS, "\n".join(outputs))

    # ---- 勾選 ----

    def _items(self) -> list[QTreeWidgetItem]:
        return [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]

    def checked_paths(self) -> list[str]:
        return [item.data(COL_NAME, _PATH) for item in self._items()
                if item.checkState(COL_NAME) == Qt.CheckState.Checked]

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing or column != COL_NAME:
            return
        self._update_count()
        self.toggled.emit([item.data(COL_NAME, _PATH)], item.checkState(COL_NAME) == Qt.CheckState.Checked)

    def _check_where(self, want: Callable[[QTreeWidgetItem], bool]) -> None:
        """勾選符合條件的、取消其餘的，一次發一個 toggled。"""
        on, off = [], []
        self._syncing = True
        for item in self._items():
            checked = want(item)
            if checked != (item.checkState(COL_NAME) == Qt.CheckState.Checked):
                item.setCheckState(COL_NAME, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                (on if checked else off).append(item.data(COL_NAME, _PATH))
        self._syncing = False
        self._update_count()
        if off:
            self.toggled.emit(off, False)
        if on:
            self.toggled.emit(on, True)

    def _set_all(self, checked: bool) -> None:
        self._check_where(lambda item: checked)

    def set_checked(self, paths: set[str], checked: bool) -> None:
        """佇列那邊加入或移除了檔案時，同步這裡的勾選（不會再發 toggled）。"""
        keys = {str(Path(p)).casefold() for p in paths}
        self._syncing = True
        for item in self._items():
            if item.data(COL_NAME, _PATH).casefold() in keys:
                item.setCheckState(COL_NAME, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._syncing = False
        self._update_count()

    def _update_count(self) -> None:
        total = self.tree.topLevelItemCount()
        self.count_label.setText(f"已選 {len(self.checked_paths())} / {total} 個" if total else "")
        for w in (self.all_button, self.new_button, self.none_button):
            w.setEnabled(bool(total))
