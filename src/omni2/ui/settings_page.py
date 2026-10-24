"""設定分頁：工作檔與模型的儲存位置、雲端語音辨識的金鑰、講者分離引擎。"""

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
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from omni2.config.settings import Settings, save_settings
from omni2.core import paths, secrets, storage
from omni2.post.diarize import DEFAULT_THRESHOLD, PYANNOTE_MODEL

# 講者區分靈敏度 → ecapa 聚類門檻（距離越小越容易分成不同人）
SENSITIVITY = [("一般（建議）", DEFAULT_THRESHOLD),
               ("較敏感：不同的人常被併成同一人時", DEFAULT_THRESHOLD - 0.1),
               ("較寬鬆：同一人常被拆成好幾人時", DEFAULT_THRESHOLD + 0.1)]


def storage_secret_path() -> str:
    """目前設定的 Google 服務帳戶金鑰路徑（存在認證管理員裡的是路徑，不是金鑰內容）。"""
    return secrets.get_secret("google") or ""


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
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.addWidget(self._storage_box())
        layout.addWidget(self._cloud_box())
        layout.addWidget(self._diarize_box())
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.refresh_usage()

    # ---- 雲端語音辨識 ----

    def _cloud_box(self) -> QGroupBox:
        c = self.settings.stt.cloud
        box = QGroupBox("雲端語音辨識（金鑰存在 Windows 認證管理員，不會寫進設定檔）")
        form = QFormLayout(box)

        self.key_edits: dict[str, QLineEdit] = {}
        self.key_status: dict[str, QLabel] = {}

        # 阿里雲 Fun-ASR
        self.region_combo = QComboBox()
        self.region_combo.addItem("國際版（新加坡，dashscope-intl）", "intl")
        self.region_combo.addItem("中國大陸版（dashscope）", "cn")
        self.region_combo.setCurrentIndex(max(0, self.region_combo.findData(c.dashscope_region)))
        self.region_combo.currentIndexChanged.connect(self._save_cloud)
        form.addRow("阿里雲 Fun-ASR 金鑰", self._key_row("dashscope", self._test_dashscope))
        form.addRow("　區域", self.region_combo)
        form.addRow("", self.key_status["dashscope"])

        # OpenAI
        self.openai_model = QComboBox()
        for model, note in (("gpt-4o-transcribe", "較準，時間軸用估計的"), ("gpt-4o-mini-transcribe", "較便宜"),
                            ("whisper-1", "有逐詞時間")):
            self.openai_model.addItem(f"{model}（{note}）", model)
        self.openai_model.setCurrentIndex(max(0, self.openai_model.findData(c.openai_model)))
        self.openai_model.currentIndexChanged.connect(self._save_cloud)
        form.addRow("OpenAI 金鑰", self._key_row("openai", self._test_openai))
        form.addRow("　模型", self.openai_model)
        form.addRow("", self.key_status["openai"])

        # Google Cloud
        self.google_path = QLineEdit(storage_secret_path())
        self.google_path.setReadOnly(True)
        self.google_path.setPlaceholderText("服務帳戶金鑰 JSON（沒有就使用 Application Default Credentials）")
        pick = QPushButton("選擇檔案…")
        pick.clicked.connect(self._pick_google_key)
        row = QHBoxLayout()
        row.addWidget(self.google_path, 1)
        row.addWidget(pick)
        self.google_project = QLineEdit(c.google_project)
        self.google_project.setPlaceholderText("例如 my-project-123")
        self.google_project.editingFinished.connect(self._save_cloud)
        self.google_location = QComboBox()
        for loc in ("us", "eu", "global", "asia-southeast1", "asia-northeast1"):
            self.google_location.addItem(loc, loc)
        self.google_location.setCurrentIndex(max(0, self.google_location.findData(c.google_location)))
        self.google_location.currentIndexChanged.connect(self._save_cloud)
        form.addRow("Google 服務帳戶金鑰", row)
        form.addRow("　專案 ID", self.google_project)
        form.addRow("　區域", self.google_location)

        note = QLabel("※ OpenAI 和 Google 目前只用模擬的回應測試過，還沒用真實金鑰驗證。")
        note.setStyleSheet("color: #666;")
        form.addRow("", note)
        return box

    def _key_row(self, name: str, test) -> QHBoxLayout:
        edit = QLineEdit()
        edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edits[name] = edit
        self.key_status[name] = QLabel()
        save = QPushButton("儲存")
        save.clicked.connect(lambda: self._save_key(name))
        delete = QPushButton("刪除")
        delete.clicked.connect(lambda: self._delete_key(name))
        check = QPushButton("測試")
        check.clicked.connect(test)
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        for b in (save, delete, check):
            row.addWidget(b)
        self._show_key_state(name)
        return row

    def _show_key_state(self, name: str) -> None:
        source = secrets.secret_source(name)
        env = secrets.ENV_VARS.get(name, "")
        text = {"keyring": "已儲存（留空表示不變更）", "env": f"使用環境變數 {env}"}.get(source, "貼上金鑰後按「儲存」")
        self.key_edits[name].setPlaceholderText(text)
        self.key_status[name].setText({"keyring": "✓ 已設定", "env": "✓ 已設定（環境變數）"}.get(source, "尚未設定"))

    def _save_key(self, name: str) -> None:
        value = self.key_edits[name].text().strip()
        if not value:
            return
        secrets.set_secret(name, value)
        self.key_edits[name].clear()
        self._show_key_state(name)

    def _delete_key(self, name: str) -> None:
        if secrets.secret_source(name) != "keyring":
            return
        if QMessageBox.question(self, "刪除金鑰", "要從 Windows 認證管理員刪除這把金鑰嗎？") \
                == QMessageBox.StandardButton.Yes:
            secrets.set_secret(name, None)
            self._show_key_state(name)

    def _save_cloud(self) -> None:
        c = self.settings.stt.cloud
        c.dashscope_region = self.region_combo.currentData()
        c.openai_model = self.openai_model.currentData()
        c.google_project = self.google_project.text().strip()
        c.google_location = self.google_location.currentData()
        save_settings(self.settings)

    def _pick_google_key(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "選擇服務帳戶金鑰", "", "JSON (*.json)")
        if not path:
            return
        try:
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Google 金鑰", f"讀不了這個金鑰檔：{e}")
            return
        secrets.set_secret("google", str(Path(path)))
        self.google_path.setText(str(Path(path)))
        if not self.google_project.text() and getattr(creds, "project_id", None):
            self.google_project.setText(creds.project_id)  # 金鑰檔裡就有專案 ID
        self._save_cloud()

    def _test_dashscope(self) -> None:
        from omni2.stt.funasr_engine import REGION_HOSTS, check_key

        self._save_key("dashscope")
        key = secrets.get_secret("dashscope")
        if not key:
            self.key_status["dashscope"].setText("尚未設定金鑰")
            return
        self.key_status["dashscope"].setText("測試中…")
        first = self.region_combo.currentData()
        order = [first] + [r for r in REGION_HOSTS if r != first]

        def work(_w):
            errors = []
            for region in order:  # 目前的區域不通就試另一區（兩區的金鑰不能互通）
                ok, message = check_key(key, region, self.settings.stt.cloud.funasr_model)
                if ok:
                    return region, ""
                errors.append(f"{region}：{message}")
            return None, "；".join(errors)

        def done(result):
            region, error = result
            if region is None:
                self.key_status["dashscope"].setText(f"✗ 失敗：{error[:200]}")
                return
            self.region_combo.setCurrentIndex(self.region_combo.findData(region))
            self.key_status["dashscope"].setText(f"✓ 金鑰有效（{self.region_combo.currentText()}）")

        self._run(work, done)

    def _test_openai(self) -> None:
        self._save_key("openai")
        key = secrets.get_secret("openai")
        if not key:
            self.key_status["openai"].setText("尚未設定金鑰")
            return
        self.key_status["openai"].setText("測試中…")

        def work(_w):
            from openai import OpenAI

            try:
                OpenAI(api_key=key, timeout=20).models.list()  # 不計費的請求
                return "✓ 金鑰有效"
            except Exception as e:  # noqa: BLE001
                return f"✗ 失敗：{type(e).__name__}: {str(e)[:200]}"

        self._run(work, self.key_status["openai"].setText)

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
        for w in (self.apply_button, self.clear_button, self.engine_combo, self.sensitivity_combo,
                  self.region_combo, self.openai_model):
            w.setEnabled(not busy)
        if not busy:
            self._sync_diarize()
