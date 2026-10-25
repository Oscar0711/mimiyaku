"""懸浮字幕：無邊框、永遠置頂、半透明、可拖曳，一陣子沒有新字幕就淡出。

每一句顯示兩行：上面小字是原文（講到一半時是灰色斜體的草稿），下面大字是譯文（逐字長出來）。
顯示幾句不固定：依視窗大小和字體大小，從最新一句往回放，放得下幾句就顯示幾句。
開啟「滑鼠穿透」後，點擊會直接落到下面的播放器；要關掉請用全域快捷鍵。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QLabel, QLayout, QSizeGrip, QVBoxLayout, QWidget

from omni2.config.settings import LiveSettings

HISTORY = 30  # 最多保留幾句；實際顯示幾句由視窗大小決定


def _shadowed(label: QLabel) -> QLabel:
    effect = QGraphicsDropShadowEffect(label)
    effect.setBlurRadius(6)
    effect.setOffset(1, 1)
    effect.setColor(QColor(0, 0, 0, 230))
    label.setGraphicsEffect(effect)
    return label


class _Entry(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.source = _shadowed(QLabel())
        self.translation = _shadowed(QLabel())
        for label in (self.source, self.translation):
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(0)
        lay.addWidget(self.source)
        lay.addWidget(self.translation)
        self.partial = False

    def height_for(self, width: int) -> int:
        """在這個寬度下（含自動換行）需要的高度。"""
        h = self.layout().totalHeightForWidth(width)
        return h if h >= 0 else self.sizeHint().height()


class SubtitleOverlay(QWidget):
    geometry_changed = Signal(list)

    def __init__(self, settings: LiveSettings):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("Mimiyaku 字幕")
        self.entries: dict[str, _Entry] = {}
        self.order: list[str] = []
        self._drag: QPoint | None = None

        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(16, 10, 16, 10)
        # 預設版面會強制視窗至少容納所有內容，字幕一多就縮不回去；
        # 放不下的句子由 _relayout 自己藏起來，所以不讓版面限制視窗大小
        self.box.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.setMinimumSize(200, 40)
        self.box.addStretch()
        self.hint = QLabel("字幕會顯示在這裡（可以拖曳移動、右下角調整大小）")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setStyleSheet("color: rgba(255,255,255,160);")
        self.box.addWidget(self.hint)
        self.grip = QSizeGrip(self)

        self.fade = QPropertyAnimation(self, b"windowOpacity", self)
        self.fade.setDuration(800)
        self.idle = QTimer(self)
        self.idle.setSingleShot(True)
        self.idle.timeout.connect(self._fade_out)

        self.apply(settings)
        if len(settings.overlay_geometry) == 4:
            self.setGeometry(*settings.overlay_geometry)
        else:
            self.resize(900, 160)
            screen = self.screen().availableGeometry()
            self.move(screen.center().x() - 450, screen.bottom() - 220)

    # ---- 外觀 ----

    def apply(self, s: LiveSettings) -> None:
        self.settings = s
        self.bg_alpha = int(255 * max(0.0, min(1.0, s.opacity)))
        self.fade_ms = int(s.fade_after_s * 1000)
        self.set_click_through(s.click_through)
        for entry in self.entries.values():
            self._style(entry)
        self._relayout()
        self.update()

    def _style(self, entry: _Entry) -> None:
        size = self.settings.font_size
        partial = entry.partial
        src_font = QFont(entry.font())
        src_font.setPointSize(max(9, int(size * 0.6)))
        src_font.setItalic(partial)
        entry.source.setFont(src_font)
        entry.source.setStyleSheet(f"color: {'#bbbbbb' if partial else '#dddddd'};")
        entry.source.setVisible(self.settings.show_source and bool(entry.source.text()))
        tr_font = QFont(entry.font())
        tr_font.setPointSize(size)
        tr_font.setBold(True)
        entry.translation.setFont(tr_font)
        entry.translation.setStyleSheet("color: white;")
        entry.translation.setVisible(bool(entry.translation.text()))

    def set_click_through(self, on: bool) -> None:
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, on)
        self.grip.setVisible(not on)
        if visible:
            self.show()  # 改視窗旗標後要重新顯示才會生效

    # ---- 內容 ----

    def set_text(self, entry_id: str, source: str | None = None, translation: str | None = None,
                 partial: bool = False) -> None:
        entry = self.entries.get(entry_id)
        if entry is None:
            entry = _Entry(self)
            self.entries[entry_id] = entry
            self.order.append(entry_id)
            self.box.addWidget(entry)
            self.hint.hide()
            while len(self.order) > HISTORY:
                self._discard(self.entries.pop(self.order.pop(0)))
        if source is not None:
            entry.source.setText(source)
            entry.partial = partial
        if translation is not None:
            entry.translation.setText(translation)
        self._style(entry)
        self._relayout()
        self._wake()

    def _relayout(self) -> None:
        """從最新一句往回放，放得下就顯示，放不下的（含更舊的）都藏起來。
        最新一句一定顯示，就算它比視窗還高。"""
        m = self.box.contentsMargins()
        width = max(50, self.width() - m.left() - m.right())
        room = self.height() - m.top() - m.bottom()
        spacing = max(0, self.box.spacing())
        used, full = 0, False
        for i, entry_id in enumerate(reversed(self.order)):
            entry = self.entries[entry_id]
            need = entry.height_for(width) + (spacing if i else 0)
            full = full or (i > 0 and used + need > room)
            entry.setVisible(not full)
            if not full:
                used += need

    def visible_ids(self) -> list[str]:
        return [i for i in self.order if not self.entries[i].isHidden()]

    def _discard(self, entry: _Entry) -> None:
        # 移出版面後、真正刪除前它仍是可見的子元件，會被畫在左上角，所以先隱藏
        self.box.removeWidget(entry)
        entry.hide()
        entry.deleteLater()

    def clear(self) -> None:
        for entry in self.entries.values():
            self._discard(entry)
        self.entries.clear()
        self.order.clear()
        self.hint.show()

    def _wake(self) -> None:
        self.fade.stop()
        self.setWindowOpacity(1.0)
        if self.fade_ms > 0:
            self.idle.start(self.fade_ms)

    def _fade_out(self) -> None:
        self.fade.setStartValue(self.windowOpacity())
        self.fade.setEndValue(0.0)
        self.fade.start()

    # ---- 繪製、拖曳 ----

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, self.bg_alpha))
        painter.drawRoundedRect(self.rect(), 12, 12)

    def resizeEvent(self, event) -> None:  # noqa: N802
        self.grip.move(self.width() - self.grip.width(), self.height() - self.grip.height())
        self._relayout()
        self.geometry_changed.emit(list(self.geometry().getRect()))

    def moveEvent(self, event) -> None:  # noqa: N802
        self.geometry_changed.emit(list(self.geometry().getRect()))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._wake()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None:
            self.move(event.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag = None
