"""檔案工作區：批次佇列、各檔進度、選項，下方是播放器。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omni2.config.settings import Settings, save_settings
from omni2.core import paths
from omni2.export.formats import ALL_FORMATS
from omni2.jobs.job import Job, JobSpec, is_url, job_id
from omni2.jobs.runner import STAGE_LABELS, FileRunner
from omni2.models.recommend import missing_for
from omni2.translate.glossary import Glossary
from omni2.ui.bridge import RunnerThread
from omni2.ui.player import PlayerPanel
from omni2.ui.terms_dialog import TermsDialog

MEDIA_FILTER = "影音檔 (*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.aac *.mp4 *.mkv *.webm *.mov *.avi *.ts);;所有檔案 (*)"
SOURCE_LANGS = [("自動偵測", None), ("日文", "ja"), ("英文", "en"), ("韓文", "ko"), ("中文", "zh")]
STT_ENGINES = [("Qwen3-ASR（本地，建議）", "qwen3-asr"), ("Whisper（本地）", "faster-whisper"),
               ("阿里雲 Fun-ASR（雲端）", "funasr"), ("OpenAI（雲端）", "openai"), ("Google（雲端）", "google")]
TARGET_LANGS = [("繁體中文", "zh-TW"), ("簡體中文", "zh-CN"), ("英文", "en"), ("日文", "ja"), ("韓文", "ko")]
_STATUS_TEXT = {"start": "", "done": "完成", "skip": "沿用斷點", "error": "失敗"}

# 佇列的欄位
COL_NAME, COL_STATE, COL_PROGRESS, COL_MESSAGE = range(4)


@dataclass
class QueueEntry:
    input: str
    id: str
    item: QTreeWidgetItem
    bar: QProgressBar
    state: str = "waiting"  # waiting | running | done | error | stopped
    outputs: list[str] = field(default_factory=list)
    spec: JobSpec | None = None  # 開始處理時用的設定；預覽要用同一份才找得到結果


class FileWorkspace(QWidget):
    status_changed = Signal(str)
    running_changed = Signal(bool)

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.entries: dict[str, QueueEntry] = {}
        self.thread: RunnerThread | None = None
        self.blocked = False
        self.setAcceptDrops(True)

        # ---- 工具列 ----
        self.add_button = QPushButton("新增檔案…")
        self.add_button.clicked.connect(self._pick_files)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("貼上影片網址後按 Enter")
        self.url_edit.returnPressed.connect(self._add_url)
        self.remove_button = QPushButton("移除")
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button = QPushButton("清除已完成")
        self.clear_button.clicked.connect(self._clear_done)
        self.start_button = QPushButton("開始")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.add_button)
        toolbar.addWidget(self.url_edit, 1)
        toolbar.addWidget(self.remove_button)
        toolbar.addWidget(self.clear_button)
        toolbar.addSpacing(16)
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.stop_button)

        # ---- 選項 ----
        self.options = self._build_options()

        # ---- 佇列 ----
        self.queue = QTreeWidget()
        self.queue.setColumnCount(4)
        self.queue.setHeaderLabels(("檔案", "狀態", "進度", "訊息"))
        self.queue.setRootIsDecorated(False)
        self.queue.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self.queue.header()
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_STATE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_PROGRESS, QHeaderView.ResizeMode.Fixed)
        header.setStretchLastSection(True)
        self.queue.setColumnWidth(COL_NAME, 280)
        self.queue.setColumnWidth(COL_PROGRESS, 160)
        self.queue.itemSelectionChanged.connect(self._preview_selected)
        self.queue.itemDoubleClicked.connect(self._open_output)
        self.hint = QLabel("把影音檔拖進來，或按「新增檔案」；也可以貼上影片網址。")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setStyleSheet("color: palette(mid); padding: 12px;")

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addLayout(toolbar)
        top_layout.addWidget(self.options)
        top_layout.addWidget(self.hint)
        top_layout.addWidget(self.queue, 1)

        self.player = PlayerPanel()
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self.player)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self._update_buttons()

    # ---- 選項 ----

    def _build_options(self) -> QGroupBox:
        fm, out = self.settings.file_mode, self.settings.output
        box = QGroupBox("選項")
        grid = QGridLayout(box)

        self.src_combo = QComboBox()
        for label, code in SOURCE_LANGS:
            self.src_combo.addItem(label, code)
        self.src_combo.setCurrentIndex(max(0, self.src_combo.findData(fm.src_lang)))
        self.translate_check = QCheckBox("翻譯成")
        self.translate_check.setChecked(fm.translate)
        self.tgt_combo = QComboBox()
        for label, code in TARGET_LANGS:
            self.tgt_combo.addItem(label, code)
        self.tgt_combo.setCurrentIndex(max(0, self.tgt_combo.findData(self.settings.translate.target)))
        self.translate_check.toggled.connect(self._sync_enabled)

        self.format_checks: dict[str, QCheckBox] = {}
        formats = QHBoxLayout()
        for name in ALL_FORMATS:
            check = QCheckBox(name.upper())
            check.setChecked(name in out.formats)
            self.format_checks[name] = check
            formats.addWidget(check)
        self.bilingual_check = QCheckBox("雙語字幕")
        self.bilingual_check.setChecked(out.bilingual)
        formats.addWidget(self.bilingual_check)
        formats.addStretch()

        self.vocals_check = QCheckBox("人聲分離（有背景音樂時）")
        self.vocals_check.setChecked(fm.separate_vocals)
        self.diarize_check = QCheckBox("講者分離，人數")
        self.diarize_check.setChecked(fm.diarize)
        self.diarize_check.toggled.connect(self._sync_enabled)
        self.speakers_spin = QSpinBox()
        self.speakers_spin.setRange(0, 20)
        self.speakers_spin.setSpecialValueText("自動")
        self.speakers_spin.setValue(fm.num_speakers)
        self.terms_check = QCheckBox("全文術語先行（翻譯前先確認專有名詞）")
        self.terms_check.setChecked(fm.terms_first)
        extras = QHBoxLayout()
        for w in (self.vocals_check, self.diarize_check, self.speakers_spin, self.terms_check):
            extras.addWidget(w)
        extras.addStretch()

        self.engine_combo = QComboBox()
        for label, engine in STT_ENGINES:
            self.engine_combo.addItem(label, engine)
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(self.settings.stt.engine)))
        self.engine_combo.setToolTip("雲端引擎需要先在「設定」分頁填入金鑰，會依使用量計費。")

        grid.addWidget(QLabel("來源語言"), 0, 0)
        grid.addWidget(self.src_combo, 0, 1)
        grid.addWidget(self.translate_check, 0, 2)
        grid.addWidget(self.tgt_combo, 0, 3)
        grid.addWidget(QLabel("　辨識引擎"), 0, 4)
        grid.addWidget(self.engine_combo, 0, 5)
        grid.setColumnStretch(6, 1)
        grid.addWidget(QLabel("輸出"), 1, 0)
        grid.addLayout(formats, 1, 1, 1, 6)
        grid.addWidget(QLabel("處理"), 2, 0)
        grid.addLayout(extras, 2, 1, 1, 6)

        self.out_combo = QComboBox()
        self.out_combo.addItem("和原檔相同的資料夾", None)
        self.out_combo.addItem("指定資料夾", "custom")
        self.out_combo.setCurrentIndex(1 if out.dir else 0)
        self.out_edit = QLineEdit(out.dir or "")
        self.out_edit.setReadOnly(True)
        self.out_edit.setPlaceholderText(f"網址輸入會存到 {paths.url_output_dir()}")
        self.out_browse = QPushButton("選擇…")
        self.out_browse.clicked.connect(self._pick_out_dir)
        self.out_combo.currentIndexChanged.connect(self._on_out_mode)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_combo)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(self.out_browse)
        grid.addWidget(QLabel("存到"), 3, 0)
        grid.addLayout(out_row, 3, 1, 1, 6)
        self._sync_enabled()
        return box

    def _on_out_mode(self) -> None:
        if self.out_combo.currentData() == "custom" and not self.out_edit.text():
            self._pick_out_dir()
        self._sync_enabled()

    def _pick_out_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "字幕要存到哪裡", self.out_edit.text())
        if folder:
            self.out_edit.setText(str(Path(folder)))
        elif not self.out_edit.text():
            self.out_combo.setCurrentIndex(0)  # 沒選資料夾就退回「和原檔相同」

    def _sync_enabled(self) -> None:
        translating = self.translate_check.isChecked()
        self.tgt_combo.setEnabled(translating)
        self.bilingual_check.setEnabled(translating)
        self.terms_check.setEnabled(translating)
        self.speakers_spin.setEnabled(self.diarize_check.isChecked())
        if hasattr(self, "out_combo"):
            custom = self.out_combo.currentData() == "custom"
            self.out_edit.setEnabled(custom)
            self.out_browse.setEnabled(custom)

    def _save_options(self) -> None:
        fm = self.settings.file_mode
        fm.src_lang = self.src_combo.currentData()
        fm.translate = self.translate_check.isChecked()
        fm.separate_vocals = self.vocals_check.isChecked()
        fm.diarize = self.diarize_check.isChecked()
        fm.num_speakers = self.speakers_spin.value()
        fm.terms_first = self.terms_check.isChecked()
        self.settings.translate.target = self.tgt_combo.currentData()
        self.settings.stt.engine = self.engine_combo.currentData()
        self.settings.output.formats = [n for n, c in self.format_checks.items() if c.isChecked()]
        self.settings.output.bilingual = self.bilingual_check.isChecked()
        custom = self.out_combo.currentData() == "custom" and self.out_edit.text()
        self.settings.output.dir = self.out_edit.text() if custom else None
        save_settings(self.settings)

    def sync_from_settings(self) -> None:
        """設定分頁改了辨識引擎（例如套用推薦）之後，更新這裡的選項。"""
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(self.settings.stt.engine)))

    def make_spec(self, input: str) -> JobSpec:
        fm, out = self.settings.file_mode, self.settings.output
        return JobSpec(
            input=input,
            src_lang=fm.src_lang,
            translate=fm.translate,
            tgt_lang=self.settings.translate.target,
            formats=list(out.formats),
            bilingual=out.bilingual and fm.translate,
            separate_vocals=fm.separate_vocals,
            diarize=fm.diarize,
            num_speakers=fm.num_speakers or None,
            terms_first=fm.terms_first and fm.translate,
            out_dir=out.dir,
        )

    # ---- 佇列 ----

    def add_inputs(self, inputs: list[str]) -> int:
        added = 0
        for raw in inputs:
            text = raw.strip()
            if not text:
                continue
            if not is_url(text) and not Path(text).is_file():
                continue
            jid = job_id(text)
            if jid in self.entries:
                continue
            name = text if is_url(text) else Path(text).name
            item = QTreeWidgetItem([name, "等待中", "", ""])
            item.setToolTip(COL_NAME, text)
            self.queue.addTopLevelItem(item)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            self.queue.setItemWidget(item, COL_PROGRESS, bar)
            self.entries[jid] = QueueEntry(text, jid, item, bar)
            added += 1
        self._update_buttons()
        return added

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "選擇影音檔", "", MEDIA_FILTER)
        self.add_inputs(files)

    def _add_url(self) -> None:
        url = self.url_edit.text().strip()
        if not is_url(url):
            QMessageBox.warning(self, "網址", "請貼上 http:// 或 https:// 開頭的網址。")
            return
        if self.add_inputs([url]):
            self.url_edit.clear()

    def _selected_entries(self) -> list[QueueEntry]:
        selected = set(map(id, self.queue.selectedItems()))
        return [e for e in self.entries.values() if id(e.item) in selected]

    def _remove_selected(self) -> None:
        for entry in self._selected_entries():
            if entry.state != "running":
                self._drop(entry)
        self._update_buttons()

    def _clear_done(self) -> None:
        for entry in list(self.entries.values()):
            if entry.state == "done":
                self._drop(entry)
        self._update_buttons()

    def _drop(self, entry: QueueEntry) -> None:
        self.queue.takeTopLevelItem(self.queue.indexOfTopLevelItem(entry.item))
        del self.entries[entry.id]

    def set_blocked(self, blocked: bool) -> None:
        """即時翻譯在進行時不能開始處理檔案（兩邊同時用顯示卡容易不夠）。"""
        self.blocked = blocked
        self._update_buttons()

    def _update_buttons(self) -> None:
        running = self.thread is not None
        pending = any(e.state != "done" for e in self.entries.values())
        self.start_button.setEnabled(not running and pending and not self.blocked)
        self.stop_button.setEnabled(running)
        self.options.setEnabled(not running)
        for w in (self.add_button, self.url_edit, self.remove_button, self.clear_button):
            w.setEnabled(not running)
        self.hint.setVisible(not self.entries)

    def _set_row(self, entry: QueueEntry, state: str | None = None, text: str | None = None,
                 progress: int | None = None, message: str | None = None) -> None:
        if state:
            entry.state = state
        if text is not None:
            entry.item.setText(COL_STATE, text)
        if progress is not None:
            entry.bar.setValue(progress)
        if message is not None:
            entry.item.setText(COL_MESSAGE, message)
            entry.item.setToolTip(COL_MESSAGE, message)
        color = {"error": Qt.GlobalColor.red, "done": Qt.GlobalColor.darkGreen}.get(entry.state)
        entry.item.setForeground(COL_STATE, color if color else self.palette().text())

    # ---- 執行 ----

    def start(self) -> None:
        if self.thread is not None:
            return
        if not any(c.isChecked() for c in self.format_checks.values()):
            QMessageBox.warning(self, "輸出格式", "請至少勾選一種輸出格式。")
            return
        self._save_options()
        todo = [e for e in self.entries.values() if e.state != "done"]
        if not todo:
            return
        fm = self.settings.file_mode
        missing = missing_for(self.settings, live=False, translate=fm.translate or fm.terms_first)
        if missing:
            QMessageBox.warning(self, "缺少模型", "\n".join(missing) + "\n\n請到「設定 → 硬體與模型」下載。")
            return
        for e in todo:
            e.spec = self.make_spec(e.input)
        specs = [e.spec for e in todo]
        for e in todo:
            self._set_row(e, state="running", text="排隊中", progress=0, message="")
        glossary = Glossary.load(paths.glossary_path())
        self.thread = RunnerThread(specs, self.settings, glossary, self)
        self.thread.event.connect(self._on_event)
        self.thread.review_requested.connect(self._on_review)
        self.thread.finished_jobs.connect(self._on_finished)
        self.thread.crashed.connect(self._on_crashed)
        self.thread.start()
        self._update_buttons()
        self.running_changed.emit(True)
        self.status_changed.emit(f"開始處理 {len(todo)} 個檔案")

    def stop(self) -> None:
        if self.thread is not None:
            self.thread.stop()
            self.stop_button.setEnabled(False)
            self.status_changed.emit("停止中…（已完成的部分會保留，下次可續作）")

    def is_running(self) -> bool:
        return self.thread is not None

    def _on_event(self, ev: dict) -> None:
        if ev["type"] == "status":
            self.status_changed.emit(ev["message"])
            return
        if ev["type"] != "stage":
            return
        entry = self.entries.get(ev["id"])
        if entry is None:
            return
        label = STAGE_LABELS.get(ev["stage"], ev["stage"])
        status = ev["status"]
        if status == "progress":
            self._set_row(entry, text=label, progress=int(ev["value"] * 100))
        elif status == "start":
            self._set_row(entry, text=label, progress=0)
        elif status == "error":
            self._set_row(entry, state="error", text=f"{label}失敗", message=ev.get("message", ""))
        elif ev["stage"] == "export" and status == "done":
            self._set_row(entry, state="done", text="完成", progress=100, message=ev.get("message", ""))
        else:
            self._set_row(entry, text=f"{label}{_STATUS_TEXT[status]}",
                          progress=100 if status in ("done", "skip") else None,
                          message=ev.get("message") or None)

    def _on_review(self, job: Job, result) -> None:
        dialog = TermsDialog(job.name, result, self)
        accepted = dialog.exec() == TermsDialog.DialogCode.Accepted
        if self.thread is not None:
            self.thread.answer_review(dialog.result_terms() if accepted else None)

    def _on_finished(self, jobs: list[Job], cancelled: bool) -> None:
        for job in jobs:
            entry = self.entries.get(job.id)
            if entry is None:
                continue
            if job.error:
                self._set_row(entry, state="error", message=job.error)
            elif job.outputs:
                entry.outputs = [str(p) for p in job.outputs]
                self._set_row(entry, state="done", text="完成", progress=100)
            elif entry.state == "running":
                self._set_row(entry, state="stopped", text="已停止", message="下次開始時會從斷點繼續")
        self._finish_thread()
        failed = sum(1 for j in jobs if j.error)
        self.status_changed.emit("已停止" if cancelled else f"全部處理完畢（{failed} 個失敗）" if failed
                                 else "全部處理完畢")
        self._preview_selected()

    def _on_crashed(self, detail: str) -> None:
        for entry in self.entries.values():
            if entry.state == "running":
                self._set_row(entry, state="error", text="失敗", message="程式錯誤，詳見對話框")
        self._finish_thread()
        QMessageBox.critical(self, "發生錯誤", detail[-2000:])

    def _finish_thread(self) -> None:
        if self.thread is not None:
            self.thread.wait()
            self.thread.deleteLater()
        self.thread = None
        self._update_buttons()
        self.running_changed.emit(False)

    # ---- 預覽 ----

    def _preview_selected(self) -> None:
        selected = self._selected_entries()
        if len(selected) != 1 or selected[0].state in ("running",):
            return
        entry = selected[0]
        runner = FileRunner([entry.spec or self.make_spec(entry.input)], self.settings,
                            glossary=Glossary.load(paths.glossary_path()))
        job = runner.jobs[0]
        result = runner.load_result(job)
        if result is None:
            self.player.clear()
            return
        transcript, translations = result
        try:
            media = job.source_path
        except KeyError:  # 網址還沒下載
            media = None
        self.player.load(job.name, media, transcript.segments, translations)

    def _open_output(self, item: QTreeWidgetItem, _col: int) -> None:
        for entry in self.entries.values():
            if entry.item is item and entry.outputs:
                os.startfile(Path(entry.outputs[0]).parent)  # noqa: S606 — 開啟使用者自己的輸出資料夾
                return

    # ---- 拖放 ----

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        if self.thread is not None:
            return
        mime = event.mimeData()
        inputs = [u.toLocalFile() if u.isLocalFile() else u.toString() for u in mime.urls()]
        if not inputs and mime.hasText():
            inputs = mime.text().split()
        self.add_inputs(inputs)
        event.acceptProposedAction()
