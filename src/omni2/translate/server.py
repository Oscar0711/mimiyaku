"""管理 llama-server 子行程。換模型＝重啟 server，關掉＝VRAM 完整釋放。"""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from pathlib import Path

import httpx

from omni2.core import paths
from omni2.workers.winjob import kill_with_parent

log = logging.getLogger(__name__)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class ServerError(RuntimeError):
    pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tail(path: Path, lines: int = 8) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


class LlamaServer:
    def __init__(
        self,
        exe: str,
        model: str,
        *,
        ctx_size: int = 8192,
        gpu_layers: int = 99,
        parallel: int = 1,
        extra_args: list[str] | None = None,
    ):
        self.exe = Path(exe) if exe else None
        self.model = Path(model) if model else None
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.parallel = parallel
        self.extra_args = extra_args or []
        self.log_path = paths.logs_dir() / "llama-server.log"
        self.proc: subprocess.Popen | None = None
        self.base_url = ""

    def start(self, timeout: float = 300.0) -> str:
        if not self.exe or not self.exe.exists():
            raise ServerError(f"找不到 llama-server：{self.exe}（設定 translate.server_exe）")
        if not self.model or not self.model.exists():
            raise ServerError(f"找不到翻譯模型：{self.model}（設定 translate.model_path）")
        port = _free_port()
        args = [
            str(self.exe), "-m", str(self.model),
            "--host", "127.0.0.1", "--port", str(port),
            "-c", str(self.ctx_size), "-ngl", str(self.gpu_layers), "-np", str(self.parallel),
            # 翻譯不需要 thinking，從 server 端關掉比只靠 request 參數可靠
            "--no-webui", "--reasoning", "off",
            *self.extra_args,
        ]
        log.info("啟動 llama-server：%s", self.model.name)
        log_file = open(self.log_path, "wb")
        try:
            self.proc = subprocess.Popen(
                args, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                cwd=self.exe.parent, creationflags=_NO_WINDOW,
            )
        finally:
            log_file.close()  # 子行程已經繼承了檔案 handle
        kill_with_parent(int(self.proc._handle))
        self.base_url = f"http://127.0.0.1:{port}"

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise ServerError(f"llama-server 啟動失敗（exit {self.proc.returncode}）：\n{_tail(self.log_path)}")
            try:
                # 載入模型期間回 503，載完才回 200
                if httpx.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    log.info("llama-server 就緒（port %d）", port)
                    return self.base_url
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise ServerError(f"llama-server 在 {timeout:.0f} 秒內沒有就緒：\n{_tail(self.log_path)}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None

    def __enter__(self) -> LlamaServer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
