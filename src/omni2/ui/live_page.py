"""即時翻譯分頁：選音源、開始／停止、即時逐字稿、懸浮字幕設定、紀錄匯出。"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omni2.config.settings import Settings, save_settings
from omni2.core import paths
from omni2.export.formats import clock
from omni2.models.recommend import missing_for
from omni2.translate.glossary import Glossary
from omni2.ui.hotkeys import HOTKEYS
from omni2.ui.overlay import SubtitleOverlay
from omni2.ui.workspace import SOURCE_LANGS, TARGET_LANGS

MODES = [("系統聲音（喇叭播放的所有聲音）", "system"), ("指定程式", "app"), ("麥克風", "mic"),
         ("麥克風＋系統聲音（線上會議）", "mic+system")]
ENGINES = [("Qwen3-ASR 0.6B（本地，建議）", "qwen3-asr"), ("Whisper turbo（本地）", "faster-whisper"),
           ("阿里雲 Fun-ASR 即時（雲端，需金鑰）", "funasr")]
PRESETS = [("一般對話（建議）", "conversation"), ("快節奏直播（切得比較碎、延遲較低）", "live"),
           ("ASMR／輕聲", "asmr")]
COL_TIME, COL_SRC, COL_TEXT, COL_TRANSLATION = range(4)


class _Bridge(QObject):
    """LiveSession 的事件來自各個執行緒，轉成 Qt signal 送回介面執行緒。"""

    event = Signal(dict)


class LivePage(QWidget):
    running_changed = Signal(bool)

    def __init__(self, settings: Settings, overlay: SubtitleOverlay, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.overlay = overlay
        self.session = None
        self.state = "idle"  # idle | starting | running | stopping
        self.blocked = False
        self.rows: dict[str, int] = {}
        self.labels: dict[str, str] = {}
        self.lat_text: list[float] = []
        self.lat_first: list[float] = []
        self.bridge = _Bridge()
        self.bridge.event.connect(self._on_event)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._source_box())
        left_layout.addWidget(self._recognition_box())
        left_layout.addWidget(self._overlay_box())
        left_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(430)

        self.start_button = QPushButton("開始即時翻譯")
        self.start_button.setMinimumHeight(36)
        self.start_button.clicked.connect(self.toggle)
        self.status = QLabel("就緒")
        self.latency = QLabel("")
        self.latency.setStyleSheet("color: #666;")
        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.status, 1)
        controls.addWidget(self.latency)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(("時間", "來源", "原文", "譯文"))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        header = self.table.horizontalHeader()
        for col, mode in ((COL_TIME, QHeaderView.ResizeMode.ResizeToContents),
                          (COL_SRC, QHeaderView.ResizeMode.ResizeToContents),
                          (COL_TEXT, QHeaderView.ResizeMode.Stretch),
                          (COL_TRANSLATION, QHeaderView.ResizeMode.Stretch)):
            header.setSectionResizeMode(col, mode)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addLayout(controls)
        right_layout.addWidget(self.table, 1)
        right_layout.addWidget(self._record_box())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self._refresh_devices()
        self._sync_mode()

    # ---- 設定區 ----

    def _source_box(self) -> QGroupBox:
        live = self.settings.live
        box = QGroupBox("音源")
        form = QFormLayout(box)
        self.mode_combo = QComboBox()
        for label, mode in MODES:
            self.mode_combo.addItem(label, mode)
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(live.source)))
        self.mode_combo.currentIndexChanged.connect(self._sync_mode)
        self.output_combo = QComboBox()
        self.mic_combo = QComboBox()
        self.app_combo = QComboBox()
        refresh = QPushButton("重新整理")
        refresh.clicked.connect(self._refresh_devices)
        app_row = QHBoxLayout()
        app_row.addWidget(self.app_combo, 1)
        app_row.addWidget(refresh)
        self.app_row = QWidget()
        self.app_row.setLayout(app_row)
        app_row.setContentsMargins(0, 0, 0, 0)
        self.levels = {"mic": QProgressBar(), "sys": QProgressBar()}
        for bar in self.levels.values():
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setMaximumHeight(8)
        form.addRow("擷取", self.mode_combo)
        form.addRow("輸出裝置", self.output_combo)
        form.addRow("程式", self.app_row)
        form.addRow("麥克風", self.mic_combo)
        form.addRow("音量（系統／程式）", self.levels["sys"])
        form.addRow("音量（麥克風）", self.levels["mic"])
        self.app_hint = QLabel("只會收到這個程式的聲音，Discord、通知音等其他程式的聲音不會被辨識。"
                               "清單只列出目前有發出聲音的程式；找不到時先讓它播放，再按「重新整理」。")
        self.app_hint.setWordWrap(True)
        self.app_hint.setStyleSheet("color: #666;")
        form.addRow(self.app_hint)
        self.form = form
        return box

    def _recognition_box(self) -> QGroupBox:
        live, t = self.settings.live, self.settings.translate
        box = QGroupBox("辨識與翻譯")
        form = QFormLayout(box)
        self.engine_combo = QComboBox()
        for label, engine in ENGINES:
            self.engine_combo.addItem(label, engine)
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(live.stt_engine)))
        self.preset_combo = QComboBox()
        for label, preset in PRESETS:
            self.preset_combo.addItem(label, preset)
        self.preset_combo.setCurrentIndex(max(0, self.preset_combo.findData(live.vad_preset)))
        self.src_combo = QComboBox()
        for label, code in SOURCE_LANGS:
            self.src_combo.addItem(label, code)
        self.src_combo.setCurrentIndex(max(0, self.src_combo.findData(live.src_lang)))
        self.translate_check = QCheckBox("翻譯成")
        self.translate_check.setChecked(live.translate)
        self.tgt_combo = QComboBox()
        for label, code in TARGET_LANGS:
            self.tgt_combo.addItem(label, code)
        self.tgt_combo.setCurrentIndex(max(0, self.tgt_combo.findData(t.target)))
        row = QHBoxLayout()
        row.addWidget(self.translate_check)
        row.addWidget(self.tgt_combo, 1)
        form.addRow("辨識引擎", self.engine_combo)
        form.addRow("情境", self.preset_combo)
        form.addRow("來源語言", self.src_combo)
        form.addRow("", row)
        return box

    def _overlay_box(self) -> QGroupBox:
        live = self.settings.live
        box = QGroupBox("懸浮字幕")
        form = QFormLayout(box)
        self.overlay_check = QCheckBox("顯示懸浮字幕")
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(self._toggle_overlay)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(12, 72)
        self.font_spin.setValue(live.font_size)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(live.opacity * 100))
        self.fade_spin = QDoubleSpinBox()
        self.fade_spin.setRange(0, 60)
        self.fade_spin.setSuffix(" 秒")
        self.fade_spin.setSpecialValueText("不淡出")
        self.fade_spin.setValue(live.fade_after_s)
        self.source_check = QCheckBox("同時顯示原文")
        self.source_check.setChecked(live.show_source)
        self.click_check = QCheckBox("滑鼠穿透（點擊會落到下面的視窗）")
        self.click_check.setChecked(live.click_through)
        for w, sig in ((self.font_spin, "valueChanged"), (self.opacity_slider, "valueChanged"),
                       (self.fade_spin, "valueChanged"), (self.source_check, "toggled"),
                       (self.click_check, "toggled")):
            getattr(w, sig).connect(self._apply_overlay)
        hint = QLabel("快捷鍵：" + "、".join(label for label, _, _ in HOTKEYS.values()))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        form.addRow(self.overlay_check)
        form.addRow("字體大小", self.font_spin)
        form.addRow("背景不透明度", self.opacity_slider)
        form.addRow("沒有新字幕時淡出", self.fade_spin)
        form.addRow(self.source_check)
        form.addRow(self.click_check)
        form.addRow(hint)
        return box

    def _record_box(self) -> QGroupBox:
        box = QGroupBox("紀錄")
        self.session_combo = QComboBox()
        refresh = QPushButton("重新整理")
        refresh.clicked.connect(self._refresh_sessions)
        export_srt = QPushButton("匯出 SRT")
        export_srt.clicked.connect(lambda: self._export(["srt"]))
        export_txt = QPushButton("匯出 TXT")
        export_txt.clicked.connect(lambda: self._export(["txt"]))
        lay = QHBoxLayout(box)
        lay.addWidget(self.session_combo, 1)
        for w in (refresh, export_srt, export_txt):
            lay.addWidget(w)
        self._refresh_sessions()
        return box

    def _refresh_devices(self) -> None:
        from omni2.audio.sources import list_audio_apps, list_microphones, list_outputs

        live = self.settings.live
        for combo, items, wanted in ((self.output_combo, list_outputs(), live.output_device),
                                     (self.mic_combo, list_microphones(), live.mic_device)):
            combo.clear()
            for d in items:
                combo.addItem(d.name, d)
            index = combo.findText(wanted) if wanted else 0
            combo.setCurrentIndex(max(0, index))
        self.app_combo.clear()
        for app in list_audio_apps():
            self.app_combo.addItem(app.name, app)
        if live.app_name:
            self.app_combo.setCurrentIndex(max(0, self.app_combo.findText(live.app_name)))

    def _sync_mode(self) -> None:
        mode = self.mode_combo.currentData()
        for widget, show in ((self.output_combo, mode in ("system", "mic+system")),
                             (self.app_row, mode == "app"), (self.app_hint, mode == "app"),
                             (self.mic_combo, mode in ("mic", "mic+system")),
                             (self.levels["mic"], mode in ("mic", "mic+system"))):
            self.form.setRowVisible(widget, show)

    # ---- 懸浮字幕 ----

    def _apply_overlay(self) -> None:
        live = self.settings.live
        live.font_size = self.font_spin.value()
        live.opacity = self.opacity_slider.value() / 100
        live.fade_after_s = self.fade_spin.value()
        live.show_source = self.source_check.isChecked()
        live.click_through = self.click_check.isChecked()
        self.overlay.apply(live)

    def _toggle_overlay(self, on: bool) -> None:
        self.overlay.setVisible(on)

    def handle_hotkey(self, name: str) -> None:
        if name == "toggle_overlay":
            self.overlay_check.setChecked(not self.overlay.isVisible())
        elif name == "toggle_click_through":
            self.click_check.setChecked(not self.click_check.isChecked())

    # ---- 開始、停止 ----

    def set_blocked(self, blocked: bool) -> None:
        """檔案工作區在處理時不能開始即時翻譯（兩邊同時用顯示卡容易不夠）。"""
        self.blocked = blocked
        self._sync_controls()

    def _sync_controls(self) -> None:
        idle = self.state == "idle"
        self.start_button.setEnabled(self.state in ("running", "starting") or (idle and not self.blocked))
        self.start_button.setText("停止" if self.state in ("running", "starting") else
                                  "停止中…" if self.state == "stopping" else "開始即時翻譯")
        for box in (self.mode_combo, self.output_combo, self.mic_combo, self.app_combo, self.engine_combo,
                    self.preset_combo, self.src_combo, self.translate_check, self.tgt_combo):
            box.setEnabled(idle)

    def toggle(self) -> None:
        if self.state == "idle":
            self.start()
        elif self.state in ("running", "starting"):
            self.stop()

    def _sources(self):
        from omni2.live.session import SourceSpec

        mode = self.mode_combo.currentData()
        specs = []
        if mode in ("mic", "mic+system"):
            specs.append(SourceSpec("mic", "mic", "我", device=self.mic_combo.currentData()))
        if mode in ("system", "mic+system"):
            specs.append(SourceSpec("system", "sys", "對方", device=self.output_combo.currentData()))
        if mode == "app":
            app = self.app_combo.currentData()
            if app is None:
                raise ValueError("請先選擇要擷取的程式（讓它播放聲音後按「重新整理」）")
            specs.append(SourceSpec("app", "sys", app.name, app=app))
        return specs

    def _save_options(self) -> None:
        live = self.settings.live
        live.source = self.mode_combo.currentData()
        live.output_device = self.output_combo.currentText()
        live.mic_device = self.mic_combo.currentText()
        live.app_name = self.app_combo.currentText()
        live.stt_engine = self.engine_combo.currentData()
        live.vad_preset = self.preset_combo.currentData()
        live.src_lang = self.src_combo.currentData()
        live.translate = self.translate_check.isChecked()
        self.settings.translate.target = self.tgt_combo.currentData()
        save_settings(self.settings)

    def sync_from_settings(self) -> None:
        """設定分頁改了辨識引擎（例如套用推薦）之後，更新這裡的選項。"""
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(self.settings.live.stt_engine)))

    def start(self) -> None:
        from omni2.live.session import LiveSession

        try:
            sources = self._sources()
        except ValueError as e:
            QMessageBox.warning(self, "即時翻譯", str(e))
            return
        self._save_options()
        live = self.settings.live
        missing = missing_for(self.settings, live=True, translate=live.translate)
        if missing:
            QMessageBox.warning(self, "缺少模型", "\n".join(missing) + "\n\n請到「設定 → 硬體與模型」下載。")
            return
        self.labels = {s.label: s.display for s in sources}
        self.table.setRowCount(0)
        self.rows.clear()
        self.lat_text.clear()
        self.lat_first.clear()
        self.latency.setText("")
        self.overlay.clear()
        if self.overlay_check.isChecked():
            self.overlay.show()
        self.session = LiveSession(self.settings, sources, Glossary.load(paths.glossary_path()),
                                   self.bridge.event.emit)
        self.state = "starting"
        self._sync_controls()
        self.running_changed.emit(True)
        self.session.start()

    def stop(self) -> None:
        if self.session is not None and self.state in ("running", "starting"):
            self.state = "stopping"
            self._sync_controls()
            self.status.setText("停止中…（把最後一句處理完）")
            self.session.stop()

    def is_running(self) -> bool:
        return self.state != "idle"

    # ---- 事件 ----

    def _on_event(self, e: dict) -> None:
        kind = e["type"]
        if kind == "status":
            self.status.setText(e["message"])
        elif kind == "ready":
            self.state = "running"
            self._sync_controls()
            self.status.setText("即時翻譯中…")
        elif kind == "level":
            bar = self.levels.get("mic" if e["src"] == "mic" else "sys")
            if bar is not None:
                bar.setValue(min(100, int(e["rms"] * 400)))
        elif kind in ("partial", "final"):
            self._show_text(e)
        elif kind == "translation":
            self._set_cell(e["id"], COL_TRANSLATION, e["text"], partial=not e["done"])
            self.overlay.set_text(e["id"], translation=e["text"])
        elif kind == "latency":
            self.lat_first.append(e["lat_first"])
            self._show_latency()
        elif kind == "error":
            self.status.setText(f"錯誤：{e['message'][:120]}")
            if e.get("fatal"):
                QMessageBox.warning(self, "即時翻譯", e["message"][-1500:])
        elif kind == "stopped":
            self.state = "idle"
            self.session = None
            self._sync_controls()
            self.running_changed.emit(False)
            self.status.setText("已停止" + ("，紀錄已儲存" if e.get("log") else ""))
            for bar in self.levels.values():
                bar.setValue(0)
            self._refresh_sessions()
            save_settings(self.settings)  # 順便存懸浮字幕的位置

    def _show_text(self, e: dict) -> None:
        partial = e["type"] == "partial"
        if e["id"] not in self.rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.rows[e["id"]] = row
            self._set_cell(e["id"], COL_TIME, clock(e["start"]))
            self._set_cell(e["id"], COL_SRC, self.labels.get(e["src"], ""))
            self.table.scrollToBottom()
        self._set_cell(e["id"], COL_TEXT, e["text"], partial=partial)
        self.overlay.set_text(e["id"], source=e["text"], partial=partial)
        if not partial and "lat_text" in e:
            self.lat_text.append(e["lat_text"])
            self._show_latency()
        self.table.resizeRowToContents(self.rows[e["id"]])

    def _set_cell(self, entry_id: str, col: int, text: str, partial: bool = False) -> None:
        row = self.rows.get(entry_id)
        if row is None:
            return
        item = self.table.item(row, col) or QTableWidgetItem()
        item.setText(text)
        font = QFont(item.font())
        font.setItalic(partial)
        item.setFont(font)
        item.setForeground(QColor("#888888") if partial else self.palette().text().color())
        self.table.setItem(row, col, item)

    def _show_latency(self) -> None:
        parts = []
        if self.lat_text:
            parts.append(f"說完→原文 {sum(self.lat_text) / len(self.lat_text):.1f} 秒")
        if self.lat_first:
            parts.append(f"→第一個譯字 {sum(self.lat_first) / len(self.lat_first):.1f} 秒")
        self.latency.setText("平均 " + "，".join(parts))

    # ---- 紀錄 ----

    def _refresh_sessions(self) -> None:
        from omni2.live.record import describe, list_sessions

        self.session_combo.clear()
        for path in list_sessions()[:50]:
            self.session_combo.addItem(describe(path), path)

    def _export(self, formats: list[str]) -> None:
        from omni2.live.record import export_session

        path: Path | None = self.session_combo.currentData()
        if path is None:
            return
        out_dir = Path(self.settings.output.dir) if self.settings.output.dir else paths.url_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            files = export_session(path, out_dir, formats, self.settings.layout)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "匯出", f"匯出失敗：{e}")
            return
        os.startfile(out_dir)  # noqa: S606 — 開啟使用者自己的輸出資料夾
        self.status.setText("已匯出：" + "、".join(f.name for f in files))
