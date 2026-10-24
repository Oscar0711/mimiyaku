"""播放器：播放原始影音，同步顯示目前這句的原文和譯文；點字幕清單可以跳到那個時間點。"""

from __future__ import annotations

import bisect
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omni2.core.types import Segment
from omni2.export.formats import clock, speaker_labels

_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".ts", ".wmv"}


def segment_at(starts: list[float], segments: list[Segment], t: float) -> int | None:
    """時間 t 落在哪一句；在兩句之間的空檔回傳 None。"""
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0 and t <= segments[i].end:
        return i
    return None


class PlayerPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments: list[Segment] = []
        self.translations: list[str | None] | None = None
        self._starts: list[float] = []
        self._current: int | None = None
        self._seeking = False

        self.audio = QAudioOutput(self)
        self.media = QMediaPlayer(self)
        self.media.setAudioOutput(self.audio)
        self.video = QVideoWidget()
        self.video.setMinimumHeight(180)
        self.media.setVideoOutput(self.video)

        self.title = QLabel("選一個已完成的檔案來預覽")
        self.title.setStyleSheet("color: palette(mid);")
        self.primary = QLabel("")
        self.primary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.primary.setWordWrap(True)
        font = QFont(self.primary.font())
        font.setPointSize(font.pointSize() + 5)
        font.setBold(True)
        self.primary.setFont(font)
        self.secondary = QLabel("")
        self.secondary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.secondary.setWordWrap(True)
        self.secondary.setStyleSheet("color: #555;")

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.media.setPosition)
        self.time_label = QLabel("00:00:00 / 00:00:00")
        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.time_label)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.title)
        left_layout.addWidget(self.video, 1)
        left_layout.addWidget(self.primary)
        left_layout.addWidget(self.secondary)
        left_layout.addLayout(controls)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(("時間", "講者", "原文", "譯文"))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setWordWrap(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._jump)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.media.positionChanged.connect(self._on_position)
        self.media.durationChanged.connect(self._on_duration)
        self.media.playbackStateChanged.connect(self._on_state)

    # ---- 載入 ----

    def load(self, title: str, media_path: Path | None, segments: list[Segment],
             translations: list[str | None] | None) -> None:
        self.media.stop()
        self.segments = segments
        self.translations = translations
        self._starts = [s.start for s in segments]
        self._current = None
        self.title.setText(title)
        labels = speaker_labels(segments)
        self.table.setRowCount(len(segments))
        self.table.setColumnHidden(1, len(labels) < 2)  # 只有一位講者時這欄沒有資訊
        self.table.setColumnHidden(3, translations is None)
        for i, seg in enumerate(segments):
            translated = translations[i] if translations else ""
            for col, text in enumerate((clock(seg.start), labels.get(seg.speaker, ""), seg.text, translated or "")):
                self.table.setItem(i, col, QTableWidgetItem(text))
        self.table.resizeRowsToContents()
        self._show(None)

        playable = media_path is not None and media_path.exists()
        self.video.setVisible(playable and media_path.suffix.lower() in _VIDEO_EXTS)
        self.play_button.setEnabled(playable)
        self.slider.setEnabled(playable)
        if playable:
            self.media.setSource(QUrl.fromLocalFile(str(media_path)))
        else:
            self.media.setSource(QUrl())
            if media_path is not None:
                self.title.setText(f"{title}（找不到原始檔，只能看字幕）")

    def clear(self) -> None:
        self.load("選一個已完成的檔案來預覽", None, [], None)

    # ---- 播放控制 ----

    def toggle(self) -> None:
        if self.media.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media.pause()
        else:
            self.media.play()

    def _jump(self, row: int, _col: int) -> None:
        if 0 <= row < len(self.segments) and self.play_button.isEnabled():
            self.media.setPosition(int(self.segments[row].start * 1000))
            if self.media.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self.media.play()
        self._show(row)

    def _on_state(self, state) -> None:
        icon = (QStyle.StandardPixmap.SP_MediaPause if state == QMediaPlayer.PlaybackState.PlayingState
                else QStyle.StandardPixmap.SP_MediaPlay)
        self.play_button.setIcon(self.style().standardIcon(icon))

    def _on_duration(self, ms: int) -> None:
        self.slider.setRange(0, ms)

    def _on_position(self, ms: int) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(ms)
        self.time_label.setText(f"{clock(ms / 1000)} / {clock(self.media.duration() / 1000)}")
        index = segment_at(self._starts, self.segments, ms / 1000)
        if index != self._current:
            self._show(index)

    def _show(self, index: int | None) -> None:
        """顯示第 index 句，並在清單裡標出來。"""
        self._current = index
        if index is None:
            self.primary.setText("")
            self.secondary.setText("")
            return
        seg = self.segments[index]
        translated = self.translations[index] if self.translations else None
        self.primary.setText(translated or seg.text)
        self.secondary.setText(seg.text if translated else "")
        self.table.blockSignals(True)
        self.table.selectRow(index)
        self.table.blockSignals(False)
        self.table.scrollToItem(self.table.item(index, 0), QAbstractItemView.ScrollHint.PositionAtCenter)
