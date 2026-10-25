"""全域快捷鍵（RegisterHotKey）。開了滑鼠穿透後點不到字幕，要靠快捷鍵關掉。

WM_HOTKEY 會送到註冊它的執行緒的訊息佇列，Qt 的原生事件過濾器收得到。
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Signal

log = logging.getLogger(__name__)

_WM_HOTKEY = 0x0312
_MOD_ALT, _MOD_CONTROL, _MOD_SHIFT, _MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x4000

# 名稱 → （說明, 修飾鍵, 按鍵）
HOTKEYS = {
    "toggle_overlay": ("Ctrl+Alt+S：顯示／隱藏字幕", _MOD_CONTROL | _MOD_ALT, ord("S")),
    "toggle_click_through": ("Ctrl+Alt+C：切換滑鼠穿透", _MOD_CONTROL | _MOD_ALT, ord("C")),
}


class GlobalHotkeys(QObject, QAbstractNativeEventFilter):
    triggered = Signal(str)

    def __init__(self, app):
        QObject.__init__(self)
        QAbstractNativeEventFilter.__init__(self)
        self.app = app
        self.ids: dict[int, str] = {}
        self.failed: list[str] = []
        if sys.platform != "win32":
            return
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        for i, (name, (label, mods, key)) in enumerate(HOTKEYS.items(), start=0x5100):
            if user32.RegisterHotKey(None, i, mods | _MOD_NOREPEAT, key):
                self.ids[i] = name
            else:
                self.failed.append(label)  # 通常是被其他程式佔用了
                log.warning("無法註冊快捷鍵 %s", label)
        app.installNativeEventFilter(self)

    def nativeEventFilter(self, event_type, message):  # noqa: N802
        if event_type in (b"windows_generic_MSG", "windows_generic_MSG"):
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == _WM_HOTKEY and msg.wParam in self.ids:
                self.triggered.emit(self.ids[msg.wParam])
                return True, 0
        return False, 0

    def unregister(self) -> None:
        if sys.platform != "win32":
            return
        user32 = ctypes.WinDLL("user32")
        for i in self.ids:
            user32.UnregisterHotKey(None, i)
        self.ids.clear()
        self.app.removeNativeEventFilter(self)
