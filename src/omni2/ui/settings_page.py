"""設定分頁。左邊是目錄，右邊一次顯示一頁：

硬體與模型、辨識與翻譯、雲端語音辨識、講者分離、儲存位置、其他（捷徑、還原預設值）。
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from omni2 import __version__
from omni2.config.settings import Settings, save_settings
from omni2.core import hardware, paths, secrets, shortcut, storage
from omni2.models import catalog
from omni2.models.recommend import reset_settings
from omni2.post.diarize import DEFAULT_THRESHOLD, PYANNOTE_MODEL
from omni2.ui.models_page import ModelsPanel
from omni2.ui.tasks import Task as _Worker
from omni2.ui.tasks import run_task

# 講者區分靈敏度 → ecapa 聚類門檻（距離越小越容易分成不同人）
SENSITIVITY = [("一般（建議）", DEFAULT_THRESHOLD),
               ("較敏感：不同的人常被併成同一人時", DEFAULT_THRESHOLD - 0.1),
               ("較寬鬆：同一人常被拆成好幾人時", DEFAULT_THRESHOLD + 0.1)]
QWEN_MODELS = [("Qwen/Qwen3-ASR-1.7B", "1.7B（最準）"), ("Qwen/Qwen3-ASR-0.6B", "0.6B（較快、省顯示記憶體）")]
WHISPER_MODELS = [("large-v3-turbo", "large-v3-turbo（建議）"), ("small", "small（CPU 用）")]
OTHER_FILE = "__browse__"


def storage_secret_path() -> str:
    """目前設定的 Google 服務帳戶金鑰路徑（存在認證管理員裡的是路徑，不是金鑰內容）。"""
    return secrets.get_secret("google") or ""


def _page(*widgets: QWidget) -> QScrollArea:
    inner = QWidget()
    lay = QVBoxLayout(inner)
    for w in widgets:
        lay.addWidget(w)
    lay.addStretch()
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setWidget(inner)
    return scroll


class SettingsPage(QWidget):
    settings_changed = Signal()  # 其他分頁要重新讀設定（例如辨識引擎）
    restart_requested = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.models = ModelsPanel(settings)
        self.models.settings_changed.connect(self.reload_from_settings)
        self.sections = [
            ("硬體與模型", self.models),
            ("辨識與翻譯", _page(self._engines_box(), self._layout_box())),
            ("雲端語音辨識", _page(self._cloud_box())),
            ("講者分離", _page(self._diarize_box())),
            ("儲存位置", _page(self._storage_box())),
            ("其他", _page(self._other_box())),
        ]
        self.nav = QListWidget()
        self.nav.setFixedWidth(150)
        self.nav.setStyleSheet("QListWidget::item { padding: 6px 8px; }")
        self.stack = QStackedWidget()
        for name, widget in self.sections:
            self.nav.addItem(name)
            self.stack.addWidget(widget)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)
        outer = QHBoxLayout(self)
        outer.addWidget(self.nav)
        outer.addWidget(self.stack, 1)
        self.refresh_usage()

    def show_section(self, name: str) -> None:
        self.nav.setCurrentRow([n for n, _ in self.sections].index(name))

    def reload_from_settings(self) -> None:
        self._sync_engines()
        self.settings_changed.emit()

    # ---- 辨識與翻譯 ----

    def _engines_box(self) -> QGroupBox:
        box = QGroupBox("模型")
        self.file_qwen = self._choice_combo(QWEN_MODELS)
        self.file_whisper = self._choice_combo(WHISPER_MODELS)
        self.live_qwen = self._choice_combo(QWEN_MODELS)
        self.live_whisper = self._choice_combo(WHISPER_MODELS)
        self.translate_model = QComboBox()
        self.live_translate_model = QComboBox()
        self.server_label = QLabel()
        self.server_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.server_label.setWordWrap(True)

        form = QFormLayout(box)
        form.addRow(QLabel("<b>檔案翻譯</b>"))
        form.addRow("Qwen3-ASR 模型", self.file_qwen)
        form.addRow("Whisper 模型", self.file_whisper)
        form.addRow("翻譯模型", self.translate_model)
        form.addRow(QLabel("<b>即時翻譯</b>（辨識和翻譯模型同時放在顯示卡上）"))
        form.addRow("Qwen3-ASR 模型", self.live_qwen)
        form.addRow("Whisper 模型", self.live_whisper)
        form.addRow("翻譯模型", self.live_translate_model)
        form.addRow(QLabel("<b>翻譯推論程式</b>"))
        form.addRow("llama-server", self.server_label)
        note = QLabel("辨識引擎在「檔案翻譯」「即時翻譯」分頁選擇。標「未下載」的模型可以到「硬體與模型」下載；"
                      "語音辨識模型沒下載的話，第一次使用時也會自動下載。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        form.addRow(note)
        self._sync_engines()
        for combo in (self.file_qwen, self.file_whisper, self.live_qwen, self.live_whisper):
            combo.currentIndexChanged.connect(self._save_engines)
        self.translate_model.activated.connect(lambda _i: self._pick_translate(self.translate_model))
        self.live_translate_model.activated.connect(lambda _i: self._pick_translate(self.live_translate_model))
        return box

    @staticmethod
    def _choice_combo(choices: list[tuple[str, str]]) -> QComboBox:
        combo = QComboBox()
        for value, label in choices:
            combo.addItem(label, value)
        return combo

    @staticmethod
    def _select(combo: QComboBox, value) -> None:
        combo.blockSignals(True)
        index = combo.findData(value)
        if index < 0 and value:  # 設定檔裡手動填的值：照樣顯示
            combo.addItem(str(value), value)
            index = combo.count() - 1
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)

    def _gguf_choices(self) -> list[tuple[str, str]]:
        """目錄裡的翻譯模型（標出沒下載的），加上資料資料夾裡其他的 GGUF。"""
        out = []
        for m in catalog.CATALOG:
            if m.kind == "hf_file":
                state = "" if catalog.is_installed(m) else "（未下載）"
                out.append((str(catalog.file_path(m)), f"{m.label}{state}　{m.note}"))
        known = {Path(p).name.casefold() for p, _ in out}
        for f in sorted(paths.models_dir().glob("*.gguf")):
            if f.name.casefold() not in known:
                out.append((str(f), f.name))
        return out

    def _fill_translate(self, combo: QComboBox, current: str, same_option: bool) -> None:
        combo.blockSignals(True)
        combo.clear()
        if same_option:
            combo.addItem("和檔案翻譯相同", "")
        for path, label in self._gguf_choices():
            combo.addItem(label, path)
        index = next((i for i in range(combo.count())
                      if combo.itemData(i) and current and Path(combo.itemData(i)) == Path(current)), -1)
        if index < 0 and current:
            combo.addItem(Path(current).name + ("" if Path(current).exists() else "（找不到檔案）"), current)
            index = combo.count() - 1
        combo.addItem("選擇其他 GGUF 檔案…", OTHER_FILE)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)

    def _sync_engines(self) -> None:
        """從設定重新填入（套用推薦、下載完模型之後也會呼叫）。"""
        s = self.settings
        self._select(self.file_qwen, s.stt.qwen_model)
        self._select(self.file_whisper, s.stt.model)
        self._select(self.live_qwen, s.live.qwen_model)
        self._select(self.live_whisper, s.live.whisper_model)
        self._fill_translate(self.translate_model, s.translate.model_path, False)
        self._fill_translate(self.live_translate_model, s.live.translate_model_path, True)
        exe = s.translate.server_exe
        if s.translate.endpoint:
            self.server_label.setText(f"使用外部服務：{s.translate.endpoint}")
        elif exe and Path(exe).exists():
            self.server_label.setText(f"{Path(exe).parent.name}　{exe}")
        else:
            self.server_label.setText("還沒有安裝，請到「硬體與模型」下載 llama.cpp")

    def _save_engines(self) -> None:
        s = self.settings
        s.stt.qwen_model = self.file_qwen.currentData()
        s.stt.model = self.file_whisper.currentData()
        s.live.qwen_model = self.live_qwen.currentData()
        s.live.whisper_model = self.live_whisper.currentData()
        s.translate.model_path = self.translate_model.currentData() or s.translate.model_path
        s.live.translate_model_path = self.live_translate_model.currentData() or ""
        save_settings(s)
        self.models.refresh()

    def _pick_translate(self, combo: QComboBox) -> None:
        if combo.currentData() == OTHER_FILE:
            path, _ = QFileDialog.getOpenFileName(self, "選擇翻譯模型", str(paths.models_dir()), "GGUF (*.gguf)")
            if path:
                if combo is self.translate_model:
                    self.settings.translate.model_path = str(Path(path))
                else:
                    self.settings.live.translate_model_path = str(Path(path))
                save_settings(self.settings)
            self._sync_engines()
            self.models.refresh()
            return
        self._save_engines()

    def _layout_box(self) -> QGroupBox:
        lay = self.settings.layout
        box = QGroupBox("字幕排版（檔案翻譯輸出的字幕）")
        self.line_width = QSpinBox()
        self.line_width.setRange(8, 40)
        self.line_width.setValue(lay.max_line_width // 2)
        self.line_width.setSuffix(" 個中文字")
        self.max_lines = QSpinBox()
        self.max_lines.setRange(1, 3)
        self.max_lines.setValue(lay.max_lines)
        self.max_lines.setSuffix(" 行")
        self.max_cps = QDoubleSpinBox()
        self.max_cps.setRange(4.0, 20.0)
        self.max_cps.setSingleStep(0.5)
        self.max_cps.setValue(lay.max_cps)
        self.max_cps.setSuffix(" 字／秒")
        self.max_cps.setToolTip("每句字幕至少顯示到讀得完的時間；英文等半形字算半個字")
        for w in (self.line_width, self.max_lines, self.max_cps):
            w.valueChanged.connect(self._save_layout)
        form = QFormLayout(box)
        form.addRow("每行最多", self.line_width)
        form.addRow("每句最多", self.max_lines)
        form.addRow("閱讀速度上限", self.max_cps)
        return box

    def _save_layout(self) -> None:
        lay = self.settings.layout
        lay.max_line_width = self.line_width.value() * 2
        lay.max_lines = self.max_lines.value()
        lay.max_cps = self.max_cps.value()
        save_settings(self.settings)

    # ---- 其他 ----

    def _other_box(self) -> QGroupBox:
        box = QGroupBox("其他")
        self.shortcut_label = QLabel()
        desktop = QPushButton("在桌面建立捷徑")
        desktop.clicked.connect(lambda: self._make_shortcut(shortcut.desktop_dir(), "桌面"))
        start_menu = QPushButton("加到開始功能表")
        start_menu.clicked.connect(lambda: self._make_shortcut(shortcut.start_menu_dir(), "開始功能表"))
        row = QHBoxLayout()
        row.addWidget(desktop)
        row.addWidget(start_menu)
        row.addStretch()

        self.reset_button = QPushButton("還原預設設定…")
        self.reset_button.clicked.connect(self._reset)
        reset_note = QLabel("金鑰、儲存位置、已下載的模型不受影響；模型會依這台電腦的硬體重新推薦。")
        reset_note.setStyleSheet("color: #666;")
        open_dir = QPushButton("開啟設定檔資料夾")
        open_dir.clicked.connect(lambda: os.startfile(paths.app_dir()))  # noqa: S606

        lay = QVBoxLayout(box)
        lay.addWidget(QLabel("<b>啟動捷徑</b>（點兩下就能開啟，不需要終端機）"))
        lay.addLayout(row)
        lay.addWidget(self.shortcut_label)
        lay.addSpacing(12)
        lay.addWidget(QLabel("<b>設定</b>"))
        for w in (self.reset_button, reset_note, open_dir):
            lay.addWidget(w, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addSpacing(12)
        lay.addWidget(QLabel(f"OmniTranslate {__version__}　設定檔：{paths.config_path()}"))
        self._show_shortcuts()
        return box

    def _show_shortcuts(self) -> None:
        places = [name for name, folder in (("桌面", shortcut.desktop_dir()), ("開始功能表", shortcut.start_menu_dir()))
                  if shortcut.shortcut_path(folder).exists()]
        self.shortcut_label.setText(f"已建立：{'、'.join(places)}" if places else "還沒有建立捷徑")

    def _make_shortcut(self, folder: Path, place: str) -> None:
        try:
            path = shortcut.create(folder)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "建立捷徑", f"建立失敗：{e}")
            return
        self._show_shortcuts()
        QMessageBox.information(self, "建立捷徑", f"已在{place}建立捷徑：\n{path}")

    def _reset(self) -> None:
        answer = QMessageBox.question(
            self, "還原預設設定",
            "要把所有設定還原成預設值嗎？\n\n金鑰、儲存位置、已下載的模型不受影響；"
            "辨識和翻譯模型會依這台電腦的硬體重新推薦。\n還原後需要重新開啟程式。")
        if answer != QMessageBox.StandardButton.Yes:
            return
        reset_settings(self.settings, self.models.hw or hardware.detect(paths.data_dir()))
        save_settings(self.settings)
        if QMessageBox.question(self, "還原預設設定", "已還原。要現在重新開啟程式嗎？") \
                == QMessageBox.StandardButton.Yes:
            self.restart_requested.emit()

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
            self._sync_engines()  # 翻譯模型的路徑跟著搬了
            self.models.refresh()
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
        return run_task(self, fn, on_done, on_progress, on_fail)

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
        """檔案或即時翻譯在處理時，不能搬移、清除、換模型或還原設定。"""
        for w in (self.apply_button, self.clear_button, self.engine_combo, self.sensitivity_combo,
                  self.region_combo, self.openai_model, self.reset_button, self.file_qwen, self.file_whisper,
                  self.live_qwen, self.live_whisper, self.translate_model, self.live_translate_model):
            w.setEnabled(not busy)
        self.models.set_busy(busy)
        if not busy:
            self._sync_diarize()
