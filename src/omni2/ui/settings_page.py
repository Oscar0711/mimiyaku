"""設定分頁：工作檔與模型的儲存位置、講者分離引擎。"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from omni2.config.settings import Settings, save_settings
from omni2.core import paths, storage
from omni2.post.diarize import DEFAULT_THRESHOLD, PYANNOTE_MODEL

# 講者區分靈敏度 → ecapa 聚類門檻（距離越小越容易分成不同人）
SENSITIVITY = [("一般（建議）", DEFAULT_THRESHOLD),
               ("較敏感：不同的人常被併成同一人時", DEFAULT_THRESHOLD - 0.1),
               ("較寬鬆：同一人常被拆成好幾人時", DEFAULT_THRESHOLD + 0.1)]


class _Worker(QThread):
    """在背景執行耗時的檔案操作（算空間、搬移）。"""

    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.cancel = threading.Event()

    def run(self) -> None:
        try:
            self.done.emit(self.fn(self))
        except storage.StorageCancelled:
            self.failed.emit("已取消。已經搬過去的檔案會留在新位置，原本的檔案沒有刪除。")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


class SettingsPage(QWidget):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        layout = QVBoxLayout(self)
        layout.addWidget(self._storage_box())
        layout.addWidget(self._diarize_box())
        layout.addStretch()
        self.refresh_usage()

    # ---- 儲存位置 ----

    def _storage_box(self) -> QGroupBox:
        box = QGroupBox("工作檔與模型的儲存位置")
        self.dir_edit = QLineEdit(str(paths.data_dir()))
        self.dir_edit.setReadOnly(True)
        browse = QPushButton("選擇資料夾…")
        browse.clicked.connect(self._pick_dir)
        row = QHBoxLayout()
        row.addWidget(self.dir_edit, 1)
        row.addWidget(browse)
        self.hf_check = QCheckBox("語音辨識模型（Hugging Face 快取）也放在這裡")
        self.hf_check.setChecked(self.settings.storage.hf_in_data_dir)
        self.hf_check.setToolTip("系統預設放在 %USERPROFILE%\\.cache\\huggingface。整個快取都會搬過來，"
                                 "其他也用這個快取的程式（例如舊版全能翻譯）會找不到這些模型。")
        # 每個項目一列：名稱｜大小｜位置。用表格排，長路徑才不會把版面擠壞。
        self.usage_grid = QGridLayout()
        self.usage_grid.setColumnStretch(2, 1)
        self.usage_label = QLabel("計算中…")
        self.apply_button = QPushButton("套用並搬移")
        self.apply_button.clicked.connect(self._apply)
        self.clear_button = QPushButton("清除工作檔…")
        self.clear_button.setToolTip("刪除中間音訊和斷點；已輸出的字幕不受影響，但之後重做會從頭開始。")
        self.clear_button.clicked.connect(self._clear_jobs)
        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch()

        note = QLabel("設定檔和術語表固定放在 %LOCALAPPDATA%\\OmniTranslate2。")
        note.setStyleSheet("color: #666;")
        lay = QVBoxLayout(box)
        lay.addLayout(row)
        lay.addWidget(self.hf_check)
        lay.addLayout(self.usage_grid)
        lay.addWidget(self.usage_label)
        lay.addLayout(buttons)
        lay.addWidget(note)
        return box

    def refresh_usage(self) -> None:
        def work(_w):
            return storage.usage(self.settings)

        def show(items):
            while self.usage_grid.count():
                old = self.usage_grid.takeAt(0)
                if old.widget():
                    old.widget().deleteLater()
            for row, item in enumerate(items):
                size = QLabel(storage.human(item.size))
                size.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                where = QLabel(str(item.path))
                where.setStyleSheet("color: #666;")
                where.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                for col, widget in enumerate((QLabel(item.label), size, where)):
                    self.usage_grid.addWidget(widget, row, col)
            self.usage_label.setText(f"合計 {storage.human(sum(i.size for i in items))}")

        self._run(work, show)

    def _pick_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "選擇儲存位置", self.dir_edit.text())
        if folder:
            self.dir_edit.setText(str(Path(folder)))

    def _apply(self) -> None:
        new_dir = Path(self.dir_edit.text())
        want_hf = self.hf_check.isChecked()
        same_dir = new_dir.resolve() == paths.data_dir().resolve()
        if same_dir and want_hf == self.settings.storage.hf_in_data_dir:
            QMessageBox.information(self, "儲存位置", "位置沒有變更。")
            return
        moving = [] if same_dir else ["工作檔", "翻譯與其他模型", "llama.cpp"]
        if want_hf != self.settings.storage.hf_in_data_dir:
            moving.append("語音辨識模型（Hugging Face 快取）" + ("搬進這個資料夾" if want_hf else "搬回系統預設位置"))
        answer = QMessageBox.question(
            self, "搬移資料",
            f"要把以下資料搬到 {new_dir} 嗎？\n\n・" + "\n・".join(moving)
            + "\n\n跨磁碟搬移可能要幾分鐘。搬移期間請不要關閉程式。")
        if answer != QMessageBox.StandardButton.Yes:
            return

        dialog = QProgressDialog("搬移中…", "取消", 0, 1000, self)
        dialog.setWindowTitle("搬移資料")
        dialog.setMinimumDuration(0)

        def work(w: _Worker):
            storage.relocate(self.settings, new_dir, want_hf,
                             progress=lambda d, t, f: w.progress.emit(d, t, f), cancel=w.cancel)

        def on_progress(done: int, total: int, name: str) -> None:
            dialog.setValue(int(1000 * done / total) if total else 1000)
            dialog.setLabelText(f"{storage.human(done)} / {storage.human(total)}\n{name}")

        def finish(_result) -> None:
            dialog.close()
            self.dir_edit.setText(str(paths.data_dir()))
            self.refresh_usage()
            QMessageBox.information(self, "搬移資料", f"完成。之後的工作檔與模型都會放在：\n{paths.data_dir()}")

        def fail(_message: str) -> None:
            dialog.close()
            self.dir_edit.setText(str(paths.data_dir()))
            self.refresh_usage()  # 可能已經搬了一部分

        worker = self._run(work, finish, on_progress, on_fail=fail)
        dialog.canceled.connect(worker.cancel.set)

    def _clear_jobs(self) -> None:
        answer = QMessageBox.question(
            self, "清除工作檔",
            "要刪除所有工作檔嗎？\n\n已輸出的字幕不受影響，但已處理過的檔案之後若要重做，會從頭開始。")
        if answer != QMessageBox.StandardButton.Yes:
            return

        def done(size: int) -> None:
            self.refresh_usage()
            QMessageBox.information(self, "清除工作檔", f"已釋放 {storage.human(size)}。")

        self._run(lambda _w: storage.clear_jobs(), done)

    def _run(self, fn, on_done, on_progress=None, on_fail=None) -> _Worker:
        worker = _Worker(fn, self)
        worker.done.connect(on_done)
        if on_progress:
            worker.progress.connect(on_progress)

        def failed(message: str) -> None:
            if on_fail:
                on_fail(message)
            QMessageBox.warning(self, "發生錯誤", message)

        worker.failed.connect(failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        return worker

    # ---- 講者分離 ----

    def _diarize_box(self) -> QGroupBox:
        d = self.settings.diarization
        box = QGroupBox("講者分離")
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("內建（不需帳號）", "ecapa")
        self.engine_combo.addItem("pyannote（較準，需要 Hugging Face 帳號）", "pyannote")
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(d.engine)))
        self.engine_combo.currentIndexChanged.connect(self._save_diarize)
        self.sensitivity_combo = QComboBox()
        for label, value in SENSITIVITY:
            self.sensitivity_combo.addItem(label, value)
        closest = min(range(len(SENSITIVITY)), key=lambda i: abs(SENSITIVITY[i][1] - d.threshold))
        self.sensitivity_combo.setCurrentIndex(closest)
        self.sensitivity_combo.currentIndexChanged.connect(self._save_diarize)

        help_text = QLabel(
            "人數已知時，在檔案翻譯的「講者分離，人數」直接指定最準。<br>"
            "使用 pyannote 前要做一次設定：<br>"
            f"1. 登入 Hugging Face，到 <a href='https://hf.co/{PYANNOTE_MODEL}'>{PYANNOTE_MODEL}</a> 同意使用條款<br>"
            "2. 到 <a href='https://hf.co/settings/tokens'>hf.co/settings/tokens</a> 建立一個 Read 權限的 token<br>"
            "3. 在專案資料夾的終端機執行 <code>uv run hf auth login</code>，貼上 token")
        help_text.setOpenExternalLinks(True)
        help_text.setWordWrap(True)
        self.login_label = QLabel("")
        check = QPushButton("檢查登入狀態")
        check.clicked.connect(self._check_login)
        login_row = QHBoxLayout()
        login_row.addWidget(check)
        login_row.addWidget(self.login_label, 1)

        form = QFormLayout()
        form.addRow("引擎", self.engine_combo)
        form.addRow("區分靈敏度（內建引擎）", self.sensitivity_combo)
        lay = QVBoxLayout(box)
        lay.addLayout(form)
        lay.addWidget(help_text)
        lay.addLayout(login_row)
        self._sync_diarize()
        return box

    def _sync_diarize(self) -> None:
        self.sensitivity_combo.setEnabled(self.engine_combo.currentData() == "ecapa")

    def _save_diarize(self) -> None:
        self.settings.diarization.engine = self.engine_combo.currentData()
        self.settings.diarization.threshold = round(self.sensitivity_combo.currentData(), 3)
        self._sync_diarize()
        save_settings(self.settings)

    def _check_login(self) -> None:
        self.login_label.setText("檢查中…")

        def work(_w):
            from huggingface_hub import whoami

            try:
                return f"已登入：{whoami()['name']}"
            except Exception:  # noqa: BLE001 — 未登入或網路錯誤
                return "尚未登入（或連不上 Hugging Face）"

        self._run(work, self.login_label.setText)

    def set_busy(self, busy: bool) -> None:
        """檔案工作區在處理時，不能搬移或清除。"""
        for w in (self.apply_button, self.clear_button, self.engine_combo, self.sensitivity_combo):
            w.setEnabled(not busy)
        if not busy:
            self._sync_diarize()
