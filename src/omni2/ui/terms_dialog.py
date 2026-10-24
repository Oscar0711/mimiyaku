"""術語確認對話框：翻譯前讓使用者檢查、修改模型抽出來的摘要和專有名詞。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from omni2.translate.terms import TermsResult

_COLUMNS = ("原文", "譯名", "說明")
_KEYS = ("source", "target", "note")


class TermsDialog(QDialog):
    def __init__(self, job_name: str, result: TermsResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"確認術語：{job_name}")
        self.resize(720, 560)
        self._types = {t["source"]: t.get("type", "") for t in result.terms}

        intro = QLabel("翻譯前先確認這份檔案的摘要和專有名詞。改好的譯名會固定用在整份字幕裡。")
        intro.setWordWrap(True)

        self.summary = QPlainTextEdit(result.summary)
        self.summary.setPlaceholderText("內容摘要（會提供給譯者當背景）")
        self.summary.setFixedHeight(90)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(1, 180)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        for term in result.terms:
            self._append(term)

        add = QPushButton("新增一列")
        add.clicked.connect(self._add_row)
        remove = QPushButton("刪除選取的列")
        remove.clicked.connect(self._remove_rows)
        edit_row = QHBoxLayout()
        edit_row.addWidget(add)
        edit_row.addWidget(remove)
        edit_row.addStretch()

        self.accept_button = QPushButton("採用並開始翻譯")
        self.accept_button.setDefault(True)
        self.accept_button.clicked.connect(self.accept)
        self.skip_button = QPushButton("這份不用術語")
        self.skip_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.skip_button)
        buttons.addWidget(self.accept_button)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(QLabel("摘要"))
        layout.addWidget(self.summary)
        layout.addWidget(QLabel(f"專有名詞（{len(result.terms)} 個）"))
        layout.addWidget(self.table, 1)
        layout.addLayout(edit_row)
        layout.addLayout(buttons)

    def _append(self, term: dict) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, key in enumerate(_KEYS):
            self.table.setItem(row, col, QTableWidgetItem(str(term.get(key, ""))))

    def _add_row(self) -> None:
        self._append({})
        row = self.table.rowCount() - 1
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))

    def _remove_rows(self) -> None:
        for row in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)

    def result_terms(self) -> TermsResult:
        terms = []
        for row in range(self.table.rowCount()):
            values = {key: (self.table.item(row, col).text().strip() if self.table.item(row, col) else "")
                      for col, key in enumerate(_KEYS)}
            if values["source"] and values["target"]:
                values["type"] = self._types.get(values["source"], "")
                terms.append(values)
        return TermsResult(summary=self.summary.toPlainText().strip(), terms=terms)

    def keyPressEvent(self, event) -> None:
        # 在表格裡按 Enter 是結束編輯，不要直接關閉對話框
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.table.hasFocus():
            return
        super().keyPressEvent(event)
