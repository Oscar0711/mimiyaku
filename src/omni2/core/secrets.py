"""雲端服務的金鑰：存在 Windows 認證管理員（keyring），沒存就讀環境變數。

金鑰只在真正呼叫服務的地方（STT 子行程）讀取，不經過設定檔、工作參數或記錄檔。
"""

from __future__ import annotations

import os

SERVICE = "OmniTranslate2"
# 名稱 → 對應的環境變數（沒有存在認證管理員時的備用來源）
ENV_VARS = {
    "dashscope": "DASHSCOPE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_APPLICATION_CREDENTIALS",  # Google 存的是服務帳戶金鑰 JSON 的路徑
}


class MissingSecret(RuntimeError):
    pass


def _keyring():
    import keyring

    return keyring


def get_secret(name: str) -> str | None:
    try:
        value = _keyring().get_password(SERVICE, name)
    except Exception:  # noqa: BLE001 — 認證管理員無法使用時改讀環境變數
        value = None
    return value or os.environ.get(ENV_VARS.get(name, "")) or None


def require_secret(name: str, label: str) -> str:
    value = get_secret(name)
    if not value:
        raise MissingSecret(f"還沒有設定 {label} 的金鑰：請到「設定」分頁填入，"
                            f"或設定環境變數 {ENV_VARS.get(name, '')}")
    return value


def set_secret(name: str, value: str | None) -> None:
    """存入認證管理員；value 為空時刪除。"""
    kr = _keyring()
    if value:
        kr.set_password(SERVICE, name, value)
        return
    try:
        kr.delete_password(SERVICE, name)
    except Exception:  # noqa: BLE001 — 本來就沒有存
        pass


def secret_source(name: str) -> str | None:
    """金鑰從哪裡來：'keyring'、'env'，或 None（沒有設定）。"""
    try:
        if _keyring().get_password(SERVICE, name):
            return "keyring"
    except Exception:  # noqa: BLE001
        pass
    return "env" if os.environ.get(ENV_VARS.get(name, "")) else None
