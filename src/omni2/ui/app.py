"""主視窗。目前有「檔案翻譯」工作區；「即時翻譯」在 M4。

    uv run omni2-gui
"""

from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QTabWidget

from omni2 import __version__
from omni2.config.settings import Settings, load_settings
from omni2.ui.settings_page import SettingsPage
from omni2.ui.workspace import FileWorkspace


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.setWindowTitle(f"OmniTranslate {__version__}")
        self.resize(1280, 860)
        self.workspace = FileWorkspace(settings)
        live = QLabel("即時翻譯會在下一個階段（M4）加入。")
        live.setStyleSheet("color: palette(mid);")
        tabs = QTabWidget()
        tabs.addTab(self.workspace, "檔案翻譯")
        tabs.addTab(live, "即時翻譯")
        self.settings_page = SettingsPage(settings)
        tabs.addTab(self.settings_page, "設定")
        self.workspace.running_changed.connect(self.settings_page.set_busy)
        self.setCentralWidget(tabs)
        self.workspace.status_changed.connect(self.statusBar().showMessage)
        self.statusBar().showMessage("就緒")

    def closeEvent(self, event) -> None:
        if self.workspace.is_running():
            answer = QMessageBox.question(
                self, "還在處理中", "要停止並離開嗎？已完成的部分會保留，下次可以從斷點繼續。")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.workspace.stop()
            if self.workspace.thread is not None:
                self.workspace.thread.wait(15000)
        self.workspace.player.media.stop()
        event.accept()


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
