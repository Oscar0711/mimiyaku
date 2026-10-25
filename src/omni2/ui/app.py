"""主視窗：檔案翻譯、即時翻譯、設定三個分頁，加上懸浮字幕和全域快捷鍵。

    uv run omni2-gui
"""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTabWidget

from omni2 import __version__
from omni2.config.settings import Settings, load_settings, save_settings
from omni2.ui.hotkeys import GlobalHotkeys
from omni2.ui.live_page import LivePage
from omni2.ui.overlay import SubtitleOverlay
from omni2.ui.settings_page import SettingsPage
from omni2.ui.workspace import FileWorkspace


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle(f"OmniTranslate {__version__}")
        self.resize(1280, 860)
        self.workspace = FileWorkspace(settings)
        self.overlay = SubtitleOverlay(settings.live)
        self.overlay.geometry_changed.connect(self._remember_overlay)
        self.live_page = LivePage(settings, self.overlay)
        self.settings_page = SettingsPage(settings)
        tabs = QTabWidget()
        tabs.addTab(self.workspace, "檔案翻譯")
        tabs.addTab(self.live_page, "即時翻譯")
        tabs.addTab(self.settings_page, "設定")
        self.setCentralWidget(tabs)

        # 檔案模式和即時模式都很吃顯示卡，一邊在跑時另一邊不能開始
        self.workspace.running_changed.connect(self.settings_page.set_busy)
        self.workspace.running_changed.connect(self.live_page.set_blocked)
        self.live_page.running_changed.connect(self.workspace.set_blocked)
        self.live_page.running_changed.connect(self.settings_page.set_busy)

        self.hotkeys = GlobalHotkeys(QApplication.instance())
        self.hotkeys.triggered.connect(self.live_page.handle_hotkey)
        self.workspace.status_changed.connect(self.statusBar().showMessage)
        self.statusBar().showMessage(
            "就緒" + (f"（有快捷鍵無法註冊，可能被其他程式佔用：{'、'.join(self.hotkeys.failed)}）"
                      if self.hotkeys.failed else ""))

    def _remember_overlay(self, rect: list) -> None:
        self.settings.live.overlay_geometry = rect

    def closeEvent(self, event) -> None:  # noqa: N802
        busy = [name for name, running in (("檔案翻譯", self.workspace.is_running()),
                                            ("即時翻譯", self.live_page.is_running())) if running]
        if busy:
            answer = QMessageBox.question(
                self, "還在處理中", f"{'、'.join(busy)}還在進行，要停止並離開嗎？已完成的部分會保留。")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if self.workspace.is_running():
            self.workspace.stop()
            if self.workspace.thread is not None:
                self.workspace.thread.wait(15000)
        if self.live_page.is_running():
            self.live_page.stop()
            self._wait(lambda: not self.live_page.is_running(), 30000)
        self.workspace.player.media.stop()
        self.hotkeys.unregister()
        self.overlay.close()
        save_settings(self.settings)
        event.accept()

    @staticmethod
    def _wait(done, timeout_ms: int) -> None:
        """等背景工作結束，期間繼續處理事件（停止的通知要靠事件迴圈送回來）。"""
        loop = QEventLoop()
        check = QTimer()
        check.timeout.connect(lambda: loop.quit() if done() else None)
        check.start(100)
        QTimer.singleShot(timeout_ms, loop.quit)
        loop.exec()


def main() -> int:
    # 用 pythonw 啟動時沒有主控台，有些套件一寫 stdout 就會當掉
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    app = QApplication(sys.argv)
    app.setApplicationName("OmniTranslate")
    window = MainWindow(load_settings())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
