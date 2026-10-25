"""主視窗：檔案翻譯、即時翻譯、設定三個分頁，加上懸浮字幕和全域快捷鍵。第一次開啟時跳出設定精靈。

    uv run mimiyaku-gui
"""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QEventLoop, QProcess, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTabWidget

from omni2 import __version__
from omni2.config.settings import Settings, load_settings, save_settings
from omni2.core import hardware, paths, shortcut
from omni2.models.recommend import recommend
from omni2.ui.hotkeys import GlobalHotkeys
from omni2.ui.live_page import LivePage
from omni2.ui.overlay import SubtitleOverlay
from omni2.ui.settings_page import SettingsPage
from omni2.ui.setup_dialog import SetupDialog, needs_setup
from omni2.ui.workspace import FileWorkspace


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle(f"Mimiyaku {__version__}")
        self.resize(1280, 860)
        self.workspace = FileWorkspace(settings)
        self.overlay = SubtitleOverlay(settings.live)
        self.overlay.geometry_changed.connect(self._remember_overlay)
        self.live_page = LivePage(settings, self.overlay)
        self.settings_page = SettingsPage(settings)
        self.tabs = tabs = QTabWidget()
        tabs.addTab(self.workspace, "檔案翻譯")
        tabs.addTab(self.live_page, "即時翻譯")
        tabs.addTab(self.settings_page, "設定")
        self.setCentralWidget(tabs)
        self.settings_page.settings_changed.connect(self.workspace.sync_from_settings)
        self.settings_page.settings_changed.connect(self.live_page.sync_from_settings)
        self.settings_page.restart_requested.connect(self.restart)

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

    def first_run(self) -> None:
        """第一次開啟：偵測硬體、推薦模型組合，讓使用者勾選要下載的模型。"""
        if self.settings.first_run_done:
            return
        hw = hardware.detect(paths.data_dir())
        rec = recommend(hw)
        self.settings.first_run_done = True
        if not needs_setup(self.settings, rec):
            save_settings(self.settings)
            return
        dialog = SetupDialog(self.settings, hw, rec, self)
        if dialog.exec():
            rec.apply(self.settings)
            self.settings_page.models.set_hardware(hw)
            self.settings_page.reload_from_settings()
            self.settings_page.models.download(dialog.chosen())
            self.tabs.setCurrentWidget(self.settings_page)
            self.settings_page.show_section("硬體與模型")
        else:
            self.statusBar().showMessage("之後可以在「設定 → 硬體與模型」下載模型、套用推薦設定。")
        save_settings(self.settings)

    def restart(self) -> None:
        if self.close():
            QProcess.startDetached(sys.executable, ["-m", "omni2.ui.app"])

    def closeEvent(self, event) -> None:  # noqa: N802
        busy = [name for name, running in (("檔案翻譯", self.workspace.is_running()),
                                            ("即時翻譯", self.live_page.is_running()),
                                            ("模型下載", self.settings_page.models.is_downloading())) if running]
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
        if self.settings_page.models.is_downloading():  # 沒下載完的部分留著，下次接著下載
            self.settings_page.models.cancel_downloads()
            self.settings_page.models.queue.wait(10000)
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
    shortcut.set_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("Mimiyaku")
    app.setWindowIcon(QIcon(str(shortcut.icon_path())))
    window = MainWindow(load_settings())
    window.show()
    QTimer.singleShot(300, window.first_run)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
