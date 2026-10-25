"""第一次開啟時的設定精靈：顯示偵測到的硬體和推薦的模型組合，勾選要下載的模型。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from omni2.config.settings import Settings
from omni2.core.hardware import Hardware
from omni2.core.storage import human
from omni2.models import catalog
from omni2.models.recommend import Recommendation


def needs_setup(settings: Settings, rec: Recommendation) -> bool:
    """設定已經是推薦的組合、模型也都下載好了，就不用跳出精靈。"""
    missing = [i for i in rec.models() if not catalog.is_installed(catalog.get(i))]
    return bool(rec.changes(settings) or missing)


class SetupDialog(QDialog):
    def __init__(self, settings: Settings, hw: Hardware, rec: Recommendation, parent=None):
        super().__init__(parent)
        self.setWindowTitle("歡迎使用 OmniTranslate")
        self.setMinimumWidth(620)
        self.hw = hw

        intro = QLabel("已經偵測了這台電腦的硬體，下面是建議的模型組合。"
                       "之後隨時可以在「設定 → 硬體與模型」重新調整或下載。")
        intro.setWordWrap(True)
        lines = [f"<b>{line}</b>" if i == 0 else line for i, line in enumerate(hw.describe())]
        lines += ["", f"<b>推薦：{rec.tier}</b>", *rec.describe()]
        lines += [f"<span style='color:#b45309'>⚠ {w}</span>" for w in rec.warnings]
        changes = rec.changes(settings)
        if changes:
            lines += ["", "<b>會調整這些設定：</b>"] + [f"・{label}：{old} → {new}" for label, old, new in changes]
        summary = QLabel("<br>".join(lines))
        summary.setWordWrap(True)
        summary.setTextFormat(Qt.TextFormat.RichText)

        self.list = QListWidget()
        for model_id in rec.models():
            m = catalog.get(model_id)
            installed = catalog.is_installed(m)
            item = QListWidgetItem(f"{m.label}　約 {human(m.size)}　{m.note}" + ("　（已下載）" if installed else ""))
            item.setData(Qt.ItemDataRole.UserRole, model_id)
            if installed:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
            self.list.addItem(item)
        self.list.itemChanged.connect(self._update_total)
        self.list.setMaximumHeight(min(220, 26 * self.list.count() + 8))
        self.total = QLabel()

        buttons = QDialogButtonBox()
        self.ok = buttons.addButton("套用並下載", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("稍後再說", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addWidget(summary)
        lay.addWidget(QLabel("要下載的模型："))
        lay.addWidget(self.list)
        lay.addWidget(self.total)
        lay.addWidget(buttons)
        self._update_total()

    def chosen(self) -> list[str]:
        out = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable and item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def _update_total(self) -> None:
        size = sum(catalog.get(i).size for i in self.chosen())
        text = f"共約 {human(size)}" if size else "不用下載"
        if self.hw.disk_free_gb:
            text += f"；資料資料夾所在磁碟剩 {self.hw.disk_free_gb:.0f} GB"
        self.total.setText(text)
        self.ok.setText("套用並下載" if size else "套用")
