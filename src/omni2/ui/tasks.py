"""在背景執行緒跑耗時的工作（算空間、搬移、偵測硬體、下載），結果用 signal 送回介面執行緒。"""

from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMessageBox, QWidget

from omni2.core import storage


class Task(QThread):
    progress = Signal(object, object, str)  # 位元組數可能超過 2 GB，不能用 int（Qt 會當成 32 位元）
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn  # fn(task) -> 結果；可以用 task.progress.emit 回報進度、檢查 task.cancel
        self.cancel = threading.Event()

    def run(self) -> None:
        try:
            self.done.emit(self.fn(self))
        except storage.StorageCancelled:
            self.failed.emit("已取消。已經搬過去的檔案會留在新位置，原本的檔案沒有刪除。")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


def run_task(parent: QWidget, fn, on_done, on_progress=None, on_fail=None, quiet: bool = False) -> Task:
    """啟動背景工作。失敗時跳出錯誤訊息（quiet＝不跳，只呼叫 on_fail）。"""
    task = Task(fn, parent)
    task.done.connect(on_done)
    if on_progress:
        task.progress.connect(on_progress)

    def failed(message: str) -> None:
        if on_fail:
            on_fail(message)
        if not quiet:
            QMessageBox.warning(parent, "發生錯誤", message)

    task.failed.connect(failed)
    task.finished.connect(task.deleteLater)
    task.start()
    return task
