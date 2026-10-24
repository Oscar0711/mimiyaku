"""在背景執行緒跑 FileRunner，把事件轉成 Qt signal 送回介面執行緒。"""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QThread, Signal

from omni2.config.settings import Settings
from omni2.jobs.job import Job, JobSpec
from omni2.jobs.runner import FileRunner
from omni2.translate.glossary import Glossary
from omni2.translate.terms import TermsResult
from omni2.workers.stage import Cancelled


class RunnerThread(QThread):
    event = Signal(dict)
    # 需要使用者確認術語：（job, TermsResult）。介面必須呼叫 answer_review 回覆。
    review_requested = Signal(object, object)
    finished_jobs = Signal(list, bool)  # （jobs, 是否被停止）
    crashed = Signal(str)

    def __init__(self, specs: list[JobSpec], settings: Settings, glossary: Glossary, parent=None):
        super().__init__(parent)
        self.cancel = threading.Event()
        self._reply: TermsResult | None = None
        self._replied = threading.Event()
        self.runner = FileRunner(specs, settings, glossary=glossary, on_event=self.event.emit,
                                 review=self._review, cancel=self.cancel)

    @property
    def jobs(self) -> list[Job]:
        return self.runner.jobs

    def run(self) -> None:
        try:
            jobs = self.runner.run()
        except Cancelled:
            self.finished_jobs.emit(self.runner.jobs, True)
        except Exception:  # noqa: BLE001 — 任何錯誤都要回報給介面，不能讓執行緒默默結束
            self.crashed.emit(traceback.format_exc())
        else:
            self.finished_jobs.emit(jobs, False)

    def stop(self) -> None:
        self.cancel.set()
        self._replied.set()

    def answer_review(self, result: TermsResult | None) -> None:
        self._reply = result
        self._replied.set()

    def _review(self, job: Job, result: TermsResult) -> TermsResult | None:
        """在背景執行緒裡呼叫：請介面跳出對話框，等使用者回覆。"""
        self._reply = result
        self._replied.clear()
        self.review_requested.emit(job, result)
        self._replied.wait()
        if self.cancel.is_set():
            raise Cancelled()
        return self._reply
