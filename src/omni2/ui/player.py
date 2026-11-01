"""播放器：播放原始影音，同步顯示目前這句的原文和譯文；點字幕清單可以跳到那個時間點。"""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QUrl
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtMultimedia import (
    QAudioBufferOutput,
    QAudioFormat,
    QAudioOutput,
    QAudioSink,
    QMediaDevices,
    QMediaPlayer,
    QtAudio,
)
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from omni2.config.settings import FileModeSettings
from omni2.core.types import Segment
from omni2.export.formats import clock, speaker_labels

_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".ts", ".wmv"}
GAINS = ((1.0, "增益 關"), (1.5, "增益 ×1.5"), (2.0, "增益 ×2"), (3.0, "增益 ×3"), (4.0, "增益 ×4"))


def segment_at(starts: list[float], segments: list[Segment], t: float) -> int | None:
    """時間 t 落在哪一句；在兩句之間的空檔回傳 None。"""
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0 and t <= segments[i].end:
        return i
    return None


def volume_level(percent: int) -> float:
    """音量滑桿（0–100）換成線性倍數；用對數刻度，滑桿前半段的變化聽起來才不會太突然。"""
    return QtAudio.convertVolume(percent / 100, QtAudio.VolumeScale.LogarithmicVolumeScale,
                                 QtAudio.VolumeScale.LinearVolumeScale)


class GainOutput(QObject):
    """把聲音放大到 100% 以上。

    QAudioOutput 的音量最大只到 1.0，所以放大時把它靜音、只留著當播放時鐘（影像同步靠它），
    改從 QAudioBufferOutput 拿解碼後的聲音，乘上倍數、截掉超出範圍的部分後自己送到喇叭。
    """

    def __init__(self, media: QMediaPlayer, parent=None):
        super().__init__(parent)
        self.media = media
        self.level = 1.0
        self._format: QAudioFormat | None = None
        self._buffers: QAudioBufferOutput | None = None
        self._sink: QAudioSink | None = None
        self._io = None
        self._next_start: int | None = None  # 下一段聲音預期的開始時間（微秒），用來發現跳轉

    @property
    def active(self) -> bool:
        return self._sink is not None

    def start(self, level: float) -> None:
        self.level = level
        if self.active:
            return
        device = QMediaDevices.defaultAudioOutput()
        self._format = device.preferredFormat()
        self._format.setSampleFormat(QAudioFormat.SampleFormat.Float)
        self._buffers = QAudioBufferOutput(self._format, self)
        self._buffers.audioBufferReceived.connect(self._on_buffer)
        self._sink = QAudioSink(device, self._format, self)
        self._sink.setBufferSize(self._format.bytesForDuration(150_000))  # 緩衝越大，聲音和畫面差越多
        self._io = self._sink.start()
        self._next_start = None
        self.media.setAudioBufferOutput(self._buffers)

    def stop(self) -> None:
        if not self.active:
            return
        self.media.setAudioBufferOutput(None)
        self._sink.stop()
        self._sink.deleteLater()
        self._buffers.deleteLater()
        self._sink = self._buffers = self._io = None

    def set_paused(self, paused: bool) -> None:
        if self.active:
            self._sink.suspend() if paused else self._sink.resume()

    def _on_buffer(self, buffer) -> None:
        if not self.active:
            return
        start = buffer.startTime()
        if self._next_start is not None and abs(start - self._next_start) > 200_000:
            # 跳轉過：丟掉還沒播完的舊位置聲音
            self._sink.stop()
            self._io = self._sink.start()
        self._next_start = start + buffer.duration()
        samples = np.frombuffer(buffer.constData(), dtype=np.float32)
        self._io.write(np.clip(samples * self.level, -1.0, 1.0).tobytes())  # 寫不下的就丟掉，免得越拖越慢


class _FullscreenWindow(QWidget):
    """全螢幕時暫放播放區的黑底視窗；按 Alt+F4 也只是離開全螢幕。"""

    def __init__(self, on_close, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.on_close = on_close
        self.setWindowTitle("預覽")
        # 不能改 palette：Windows 11 樣式的播放、音量圖示是照 palette 畫的，改了會看不見
        self.setObjectName("playerFullscreen")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet("#playerFullscreen { background: black; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)

    def closeEvent(self, event) -> None:
        if event.spontaneous():  # 使用者按的；程式結束時送來的關閉要放行，不然會擋住結束
            event.ignore()
            self.on_close()


class PlayerPanel(QWidget):
    def __init__(self, settings: FileModeSettings | None = None, parent=None):
        super().__init__(parent)
        self.settings = settings or FileModeSettings()
        self.segments: list[Segment] = []
        self.translations: list[str | None] | None = None
        self._starts: list[float] = []
        self._current: int | None = None
        self._seeking = False

        self.audio = QAudioOutput(self)
        self.media = QMediaPlayer(self)
        self.media.setAudioOutput(self.audio)
        self.gain_output = GainOutput(self.media, self)
        self.video = QVideoWidget()
        self.video.setMinimumHeight(180)
        self.video.installEventFilter(self)
        self.media.setVideoOutput(self.video)

        self.title = QLabel("選一個已完成的檔案來預覽")
        self.title.setStyleSheet("color: #888;")
        self.primary = QLabel("")
        self.primary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.primary.setWordWrap(True)
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

        self.mute_button = QToolButton()
        self.mute_button.setCheckable(True)
        self.mute_button.setToolTip("靜音")
        self.mute_button.toggled.connect(self._apply_volume)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(max(0, min(100, self.settings.player_volume)))
        self.volume_slider.setFixedWidth(110)
        self.volume_slider.valueChanged.connect(self._apply_volume)
        self.volume_label = QLabel()
        self.volume_label.setMinimumWidth(self.volume_label.fontMetrics().horizontalAdvance("400%") + 4)
        self.gain_combo = QComboBox()
        for gain, label in GAINS:
            self.gain_combo.addItem(label, gain)
        self.gain_combo.setToolTip("聲音太小時放大到 100% 以上。每次開新的檔案都會回到「關」，避免突然太大聲。")
        self.gain_combo.currentIndexChanged.connect(self._apply_volume)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(8, 96)
        self.size_spin.setPrefix("字幕 ")
        self.size_spin.setSuffix(" pt")
        self.size_spin.setToolTip("預覽字幕的大小；一般和全螢幕各記一個")
        self.size_spin.valueChanged.connect(self._on_size_changed)
        self.fullscreen_button = QToolButton()
        self.fullscreen_button.setText("全螢幕")
        self.fullscreen_button.setToolTip("全螢幕（F11；Esc 離開，也可以在影片上按兩下）")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        extras = QHBoxLayout()
        extras.addWidget(self.mute_button)
        extras.addWidget(self.volume_slider)
        extras.addWidget(self.volume_label)
        extras.addWidget(self.gain_combo)
        extras.addStretch(1)
        extras.addWidget(self.size_spin)
        extras.addWidget(self.fullscreen_button)

        self.stage = QWidget()
        left_layout = QVBoxLayout(self.stage)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.title)
        left_layout.addStretch(0)
        left_layout.addWidget(self.video, 1)
        left_layout.addWidget(self.primary)
        left_layout.addWidget(self.secondary)
        left_layout.addStretch(0)
        left_layout.addLayout(controls)
        left_layout.addLayout(extras)
        self._stage_layout = left_layout
        for keys, slot in (("F11", self.toggle_fullscreen), ("Esc", self.exit_fullscreen), ("Space", self.toggle)):
            shortcut = QShortcut(QKeySequence(keys), self.stage, slot)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

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

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.stage)
        self.splitter.addWidget(self.table)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)
        self._fullscreen: _FullscreenWindow | None = None
        self._splitter_sizes: list[int] = []

        self.media.positionChanged.connect(self._on_position)
        self.media.durationChanged.connect(self._on_duration)
        self.media.playbackStateChanged.connect(self._on_state)
        self._apply_volume()
        self._apply_subtitle_style()

    # ---- 載入 ----

    def load(self, title: str, media_path: Path | None, segments: list[Segment],
             translations: list[str | None] | None) -> None:
        self.media.stop()
        self.gain_combo.setCurrentIndex(0)  # 增益不沿用上一個檔案，免得新檔案一播就太大聲
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
        self._set_video_visible(playable and media_path.suffix.lower() in _VIDEO_EXTS)
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

    def stop(self) -> None:
        """關閉程式前呼叫：停止播放、收回全螢幕視窗。"""
        self.exit_fullscreen()
        self.media.stop()
        self.gain_output.stop()

    def _set_video_visible(self, visible: bool) -> None:
        self.video.setVisible(visible)
        # 沒有影像時讓字幕置中，不要擠在上面
        for index in (1, 5):
            self._stage_layout.setStretch(index, 0 if visible else 1)

    # ---- 播放控制 ----

    def toggle(self) -> None:
        if not self.play_button.isEnabled():
            return
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
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        icon = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.play_button.setIcon(self.style().standardIcon(icon))
        self.gain_output.set_paused(not playing)

    def _on_duration(self, ms: int) -> None:
        self.slider.setRange(0, ms)

    def _on_position(self, ms: int) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(ms)
        self.time_label.setText(f"{clock(ms / 1000)} / {clock(self.media.duration() / 1000)}")
        index = segment_at(self._starts, self.segments, ms / 1000)
        if index != self._current:
            self._show(index)

    # ---- 音量 ----

    def _apply_volume(self, *_args) -> None:
        percent = self.volume_slider.value()
        gain = self.gain_combo.currentData() or 1.0
        muted = self.mute_button.isChecked()
        self.settings.player_volume = percent
        level = volume_level(percent)
        if gain > 1.0:
            self.audio.setMuted(True)  # 只當時鐘，聲音改由 GainOutput 放大後送出
            self.gain_output.start(0.0 if muted else level * gain)
            self.gain_output.set_paused(self.media.playbackState() != QMediaPlayer.PlaybackState.PlayingState)
        else:
            self.gain_output.stop()
            self.audio.setMuted(muted)
            self.audio.setVolume(level)
        self.volume_label.setText(f"{round(percent * gain)}%")
        self.volume_label.setStyleSheet("color: #d9822b; font-weight: bold;" if gain > 1.0 else "")
        # 用文字不用 QStyle 圖示：Windows 11 樣式的音量圖示移到全螢幕視窗後會畫不出來
        self.mute_button.setText("🔇" if muted or percent == 0 else "🔊")
        self.mute_button.setToolTip("取消靜音" if muted else "靜音")

    # ---- 字幕大小、全螢幕 ----

    def is_fullscreen(self) -> bool:
        return self._fullscreen is not None and self._fullscreen.isVisible()

    def _on_size_changed(self, size: int) -> None:
        if self.is_fullscreen():
            self.settings.subtitle_size_fullscreen = size
        else:
            self.settings.subtitle_size = size
        self._apply_subtitle_style()

    def _apply_subtitle_style(self) -> None:
        full = self.is_fullscreen()
        size = self.settings.subtitle_size_fullscreen if full else self.settings.subtitle_size
        self.size_spin.blockSignals(True)
        self.size_spin.setValue(size)
        self.size_spin.blockSignals(False)
        font = QFont(self.primary.font())
        font.setPointSize(size)
        font.setBold(True)
        self.primary.setFont(font)
        font = QFont(self.secondary.font())
        font.setPointSize(max(6, round(size * 0.75)))
        self.secondary.setFont(font)
        self.primary.setStyleSheet("color: white;" if full else "")
        self.secondary.setStyleSheet("color: #bbb;" if full else "color: #555;")
        self.time_label.setStyleSheet("color: #ddd;" if full else "")
        self.title.setVisible(not full)
        self.fullscreen_button.setText("離開全螢幕" if full else "全螢幕")

    def toggle_fullscreen(self) -> None:
        if self.is_fullscreen():
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def enter_fullscreen(self) -> None:
        if self.is_fullscreen():
            return
        if self._fullscreen is None:
            self._fullscreen = _FullscreenWindow(self.exit_fullscreen, self)
        self._splitter_sizes = self.splitter.sizes()
        self._fullscreen.layout().addWidget(self.stage)
        self.stage.show()
        self._fullscreen.setGeometry(self.screen().geometry())  # 開在主視窗所在的螢幕
        self._fullscreen.showFullScreen()
        self._fullscreen.activateWindow()
        self.stage.setFocus()
        self._apply_subtitle_style()

    def exit_fullscreen(self) -> None:
        if not self.is_fullscreen():
            return
        self.splitter.insertWidget(0, self.stage)
        self.stage.show()
        if self._splitter_sizes:
            self.splitter.setSizes(self._splitter_sizes)
        self._fullscreen.hide()
        self.window().activateWindow()
        self._apply_subtitle_style()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.video and event.type() == QEvent.Type.MouseButtonDblClick:
            self.toggle_fullscreen()
            return True
        return super().eventFilter(watched, event)

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
