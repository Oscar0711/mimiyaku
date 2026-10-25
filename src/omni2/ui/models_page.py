"""設定 → 硬體與模型：偵測到的硬體、推薦的模型組合、每個模型的下載狀態。

下載排成一個佇列，一次下載一個（在子行程裡），可以個別取消；沒下載完的部分下次會接著下載。
"""

from __future__ import annotations

import shutil
import threading
import time

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omni2.config.settings import Settings, save_settings
from omni2.core import hardware, paths
from omni2.core.storage import human
from omni2.models import catalog, download, recommend
from omni2.ui.tasks import run_task
from omni2.workers.stage import Cancelled

COLUMNS = ["類別", "模型", "說明", "大小", "狀態", ""]


class DownloadQueue(QThread):
    # （模型, 已下載, 總共, 說明）。位元組數用 object：宣告成 int 會被 Qt 當成 32 位元，超過 2 GB 就變負數
    progress = Signal(str, object, object, str)
    item_done = Signal(str, object)
    item_failed = Signal(str, str)

    def __init__(self, driver_cuda: float, parent=None):
        super().__init__(parent)
        self.driver_cuda = driver_cuda
        self.queue: list[str] = []
        self.current = ""
        self.cancel_current = threading.Event()
        self.stopping = False

    def add(self, ids: list[str]) -> None:
        for i in ids:
            if i != self.current and i not in self.queue:
                self.queue.append(i)

    def drop(self, model_id: str) -> None:
        """取消一個：還在排隊就移除，正在下載就中斷它。"""
        if model_id == self.current:
            self.cancel_current.set()
        elif model_id in self.queue:
            self.queue.remove(model_id)

    def stop(self) -> None:
        self.stopping = True
        self.queue.clear()
        self.cancel_current.set()

    def run(self) -> None:
        while self.queue and not self.stopping:
            model_id = self.current = self.queue.pop(0)
            self.cancel_current.clear()
            try:
                result = download.download(model_id, lambda d, t, x: self.progress.emit(model_id, d, t, x),
                                           self.cancel_current, self.driver_cuda)
            except Cancelled:
                self.item_failed.emit(model_id, "已取消")
            except Exception as e:  # noqa: BLE001
                self.item_failed.emit(model_id, str(e) or type(e).__name__)
            else:
                self.item_done.emit(model_id, result)
        self.current = ""


class ModelsPanel(QWidget):
    settings_changed = Signal()  # 套用推薦或更新 llama.cpp 之後，其他頁面要重新讀設定

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.hw: hardware.Hardware | None = None
        self.rec: recommend.Recommendation | None = None
        self.queue: DownloadQueue | None = None
        self.state: dict[str, str] = {}  # 下載中的狀態文字（排隊中、35%、失敗…）
        self.rows: dict[str, int] = {}
        self.busy = False
        self._speed: tuple[float, int, float] | None = None  # （時間, 位元組, 每秒位元組）

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._hardware_box())
        layout.addWidget(self._models_box(), 1)
        self.refresh()
        self.detect()

    # ---- 硬體與推薦 ----

    def _hardware_box(self) -> QGroupBox:
        box = QGroupBox("這台電腦與推薦的模型組合")
        self.hw_label = QLabel("偵測中…")
        self.hw_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.rec_label = QLabel()
        self.rec_label.setWordWrap(True)
        self.rec_label.setTextFormat(Qt.TextFormat.RichText)
        self.detect_button = QPushButton("重新偵測")
        self.detect_button.clicked.connect(self.detect)
        self.apply_button = QPushButton("套用推薦設定…")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_recommendation)
        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.detect_button)
        buttons.addStretch()
        lay = QVBoxLayout(box)
        lay.addWidget(self.hw_label)
        lay.addWidget(self.rec_label)
        lay.addLayout(buttons)
        return box

    def detect(self) -> None:
        self.detect_button.setEnabled(False)

        def done(hw: hardware.Hardware) -> None:
            self.detect_button.setEnabled(True)
            self.set_hardware(hw)

        def fail(_message: str) -> None:
            self.detect_button.setEnabled(True)
            self.hw_label.setText("偵測失敗")

        run_task(self, lambda _t: hardware.detect(paths.data_dir()), done, on_fail=fail, quiet=True)

    def set_hardware(self, hw: hardware.Hardware) -> None:
        self.hw = hw
        self.rec = recommend.recommend(hw)
        self.hw_label.setText("\n".join(hw.describe()))
        lines = [f"<b>推薦：{self.rec.tier}</b>"] + self.rec.describe()
        lines += [f"<span style='color:#b45309'>⚠ {w}</span>" for w in self.rec.warnings]
        if not self.rec.changes(self.settings):
            lines.append("<span style='color:#15803d'>✓ 目前的設定就是推薦的組合</span>")
        self.rec_label.setText("<br>".join(lines))
        self.apply_button.setEnabled(not self.busy)

    def apply_recommendation(self) -> None:
        if self.rec is None:
            return
        changes = self.rec.changes(self.settings)
        missing = [i for i in self.rec.models() if not catalog.is_installed(catalog.get(i))]
        if not changes and not missing:
            QMessageBox.information(self, "套用推薦設定", "目前的設定已經是推薦的組合，需要的模型也都下載好了。")
            return
        parts = []
        if changes:
            parts.append("會調整這些設定：\n" + "\n".join(f"・{label}：{old} → {new}" for label, old, new in changes))
        if missing:
            size = sum(catalog.get(i).size for i in missing)
            names = "、".join(catalog.get(i).label for i in missing)
            parts.append(f"還要下載：{names}（約 {human(size)}）")
        if QMessageBox.question(self, "套用推薦設定", "\n\n".join(parts) + "\n\n要套用嗎？") \
                != QMessageBox.StandardButton.Yes:
            return
        self.rec.apply(self.settings)
        save_settings(self.settings)
        self.settings_changed.emit()
        self.set_hardware(self.hw)
        self.download(missing)

    # ---- 模型清單 ----

    def _models_box(self) -> QGroupBox:
        box = QGroupBox("模型（粗體＝目前設定會用到）")
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = self.table.horizontalHeader()
        for col in range(len(COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.dl_label = QLabel()
        self.cancel_all = QPushButton("全部取消")
        self.cancel_all.setVisible(False)
        self.cancel_all.clicked.connect(self.cancel_downloads)
        bottom = QHBoxLayout()
        bottom.addWidget(self.dl_label, 1)
        bottom.addWidget(self.cancel_all)
        note = QLabel("語音辨識模型放在 Hugging Face 快取，其他放在資料資料夾（位置可以在「儲存位置」更改）。"
                      "取消或中斷的下載，下次會從中斷的地方接著下載。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        lay = QVBoxLayout(box)
        lay.addWidget(self.table, 1)
        lay.addLayout(bottom)
        lay.addWidget(note)
        return box

    def refresh(self) -> None:
        in_use = recommend.models_in_use(self.settings)
        self.table.setRowCount(len(catalog.CATALOG))
        self.rows.clear()
        for row, m in enumerate(catalog.CATALOG):
            self.rows[m.id] = row
            installed = catalog.is_installed(m)
            version = catalog.installed_version(m)
            name = QTableWidgetItem(m.label + (f"（{version}）" if version else ""))
            if m.id in in_use:
                font = QFont(name.font())
                font.setBold(True)
                name.setFont(font)
            size = catalog.installed_size(m) if installed else m.size
            cells = [catalog.GROUPS[m.group], name, m.note, human(size) if installed else f"約 {human(size)}",
                     self._status_text(m, installed)]
            for col, cell in enumerate(cells):
                self.table.setItem(row, col, cell if isinstance(cell, QTableWidgetItem) else QTableWidgetItem(cell))
            old = self.table.cellWidget(row, 5)
            if old is not None:  # 換掉的按鈕要等事件迴圈才會刪除，期間會畫在錯的位置，先藏起來
                old.hide()
            self.table.setCellWidget(row, 5, self._actions(m, installed))
        self.table.resizeRowsToContents()

    def _status_text(self, m: catalog.ModelInfo, installed: bool) -> str:
        if m.id in self.state:
            return self.state[m.id]
        if m.kind == "llama" and len(catalog.llama_versions()) > 1:
            return f"✓ 已安裝（另有 {len(catalog.llama_versions()) - 1} 個舊版）"
        return "✓ 已下載" if installed else "未下載"

    def _actions(self, m: catalog.ModelInfo, installed: bool) -> QWidget:
        cell = QWidget()
        lay = QHBoxLayout(cell)
        lay.setContentsMargins(4, 0, 4, 0)

        def button(text: str, slot, enabled: bool = True) -> None:
            b = QPushButton(text)
            b.clicked.connect(slot)
            b.setEnabled(enabled)
            lay.addWidget(b)

        active = self.queue is not None and (m.id == self.queue.current or m.id in self.queue.queue)
        if active:
            button("取消", lambda: self.queue and self.queue.drop(m.id))
        elif m.kind == "llama":
            button("更新" if installed else "下載", lambda: self.download([m.id]))
            if len(catalog.llama_versions()) > 1:
                button("清除舊版", self._delete_old_llama, not self.busy)
            if installed:
                button("刪除", lambda: self._delete(m), not self.busy)
        elif installed:
            button("刪除", lambda: self._delete(m), not self.busy)
        else:
            button("下載", lambda: self.download([m.id]))
        lay.addStretch()
        return cell

    def _delete(self, m: catalog.ModelInfo) -> None:
        in_use = m.id in recommend.models_in_use(self.settings)
        warn = "\n\n目前的設定會用到這個模型，刪除後要重新下載才能使用。" if in_use else ""
        size = human(catalog.installed_size(m))
        if QMessageBox.question(self, "刪除模型", f"要刪除 {m.label}（{size}）嗎？{warn}") \
                != QMessageBox.StandardButton.Yes:
            return
        catalog.delete(m)
        self.state.pop(m.id, None)
        self.refresh()
        self.settings_changed.emit()

    def _delete_old_llama(self) -> None:
        freed = catalog.delete_old_llama()
        self.refresh()
        QMessageBox.information(self, "llama.cpp", f"已刪除舊版，釋放 {human(freed)}。")

    # ---- 下載 ----

    def download(self, ids: list[str]) -> None:
        ids = [i for i in ids if i not in self.state or not self.state[i].startswith(("排隊", "下載"))]
        if not ids:
            return
        if not self._enough_space(ids):
            return
        for i in ids:
            self.state[i] = "排隊中"
        if self.queue is None:
            self.queue = DownloadQueue(self.hw.driver_cuda if self.hw else self._driver_cuda(), self)
            self.queue.progress.connect(self._on_progress)
            self.queue.item_done.connect(self._on_done)
            self.queue.item_failed.connect(self._on_failed)
            self.queue.finished.connect(self._on_queue_finished)
            self.queue.add(ids)
            self.queue.start()
        else:
            self.queue.add(ids)
        self.cancel_all.setVisible(True)
        self.refresh()
        self._update_label()

    @staticmethod
    def _driver_cuda() -> float:
        return hardware.detect().driver_cuda

    def _enough_space(self, ids: list[str]) -> bool:
        need: dict[str, int] = {}
        for i in ids:
            m = catalog.get(i)
            where = catalog.hub_cache() if m.kind == "hf_repo" else paths.data_dir()
            where.mkdir(parents=True, exist_ok=True)
            drive = where.resolve().drive or str(where)
            need[drive] = need.get(drive, 0) + m.size
        short = []
        for drive, size in need.items():
            free = shutil.disk_usage(drive + "\\").free
            if size * 1.1 > free:
                short.append(f"{drive} 需要約 {human(size)}，只剩 {human(free)}")
        if short:
            QMessageBox.warning(self, "磁碟空間不足", "\n".join(short)
                                + "\n\n可以到「儲存位置」把模型搬到空間比較大的磁碟。")
        return not short

    def is_downloading(self) -> bool:
        return self.queue is not None and self.queue.isRunning()

    def cancel_downloads(self) -> None:
        if self.queue is not None:
            for i in [self.queue.current, *self.queue.queue]:
                self.state[i] = "已取消"
            self.queue.stop()

    def _on_progress(self, model_id: str, done: int, total: int, text: str) -> None:
        now = time.monotonic()
        if self._speed is None or self._speed[1] > done:
            self._speed = (now, done, 0.0)
        elif now - self._speed[0] >= 1.0:
            rate = (done - self._speed[1]) / (now - self._speed[0])
            self._speed = (now, done, rate if not self._speed[2] else 0.7 * self._speed[2] + 0.3 * rate)
        if total:
            status = f"下載中 {100 * done // total}%"
        else:
            status = text or "處理中…"
        self.state[model_id] = status
        row = self.rows.get(model_id)
        if row is not None and self.table.item(row, 4):
            self.table.item(row, 4).setText(status)
        self._update_label(model_id, done, total)

    def _update_label(self, model_id: str = "", done: int = 0, total: int = 0) -> None:
        if self.queue is None:
            return
        waiting = len(self.queue.queue)
        current = model_id or self.queue.current
        if not current:
            self.dl_label.setText("")
            return
        text = f"正在下載 {catalog.get(current).label}"
        if total:
            text += f"：{human(done)} / {human(total)}"
            if self._speed and self._speed[2] > 0:
                text += f"（{human(int(self._speed[2]))}/s，約剩 {_eta((total - done) / self._speed[2])}）"
        if waiting:
            text += f"，還有 {waiting} 個排隊"
        self.dl_label.setText(text)

    def _on_done(self, model_id: str, result: dict) -> None:
        self.state.pop(model_id, None)
        self._speed = None
        if model_id == "llama-server" and result.get("path"):
            self.settings.translate.server_exe = result["path"]
            save_settings(self.settings)
        self.refresh()
        self.settings_changed.emit()  # 其他頁面的模型清單和提示（例如韓文微調版下載好了）跟著更新

    def _on_failed(self, model_id: str, message: str) -> None:
        self._speed = None
        self.state[model_id] = "已取消" if message == "已取消" else f"✗ 失敗：{message[:120]}"
        self.refresh()

    def _on_queue_finished(self) -> None:
        self.queue.deleteLater()
        self.queue = None
        self.cancel_all.setVisible(False)
        self.dl_label.setText("")
        # 失敗和取消的訊息留著給使用者看，其餘的清掉
        self.state = {k: v for k, v in self.state.items() if v.startswith(("✗", "已取消"))}
        self.refresh()

    def set_busy(self, busy: bool) -> None:
        """檔案或即時翻譯在跑時，不能刪除模型或改設定（下載可以）。"""
        self.busy = busy
        self.apply_button.setEnabled(not busy and self.rec is not None)
        self.refresh()


def _eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    if seconds < 3600:
        return f"{seconds / 60:.0f} 分鐘"
    return f"{seconds / 3600:.1f} 小時"
