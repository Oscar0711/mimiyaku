"""下載模型。

在子行程裡下載（omni2.workers.stage.run_stage）：取消＝直接結束子行程，主程式不會卡在網路讀取上。
沒下載完的部分留著（*.part 或 Hugging Face 的 *.incomplete），下次從斷掉的地方接著下載。
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable

from omni2.models import catalog
from omni2.models.catalog import ModelInfo
from omni2.workers.stage import run_stage

ProgressFn = Callable[[int, int, str], None]  # （已下載位元組, 總位元組, 說明）
RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"


def download(model_id: str, on_progress: ProgressFn | None = None, cancel: threading.Event | None = None,
             driver_cuda: float = 0.0) -> dict:
    """下載一個模型，回傳 {"path": 位置, "version": 版本（llama.cpp）}。"""
    def on_event(event: dict) -> None:
        if event.get("type") == "progress" and on_progress:
            on_progress(event["done"], event["total"], event.get("text", ""))

    payload = {"id": model_id, "driver_cuda": driver_cuda}
    return run_stage("omni2.models.download:worker", payload, on_event, cancel)


# ---- 以下在子行程裡執行 ----

class _Progress:
    def __init__(self, emit, total: int):
        self.emit = emit
        self.total = total
        self.last = 0.0

    def __call__(self, done: int, text: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self.last >= 0.25:
            self.last = now
            self.emit({"type": "progress", "done": done, "total": self.total, "text": text})


def worker(payload: dict, emit) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    m = catalog.get(payload["id"])
    if m.kind == "hf_repo":
        return _hf_repo(m, emit)
    if m.kind == "hf_file":
        from huggingface_hub import hf_hub_url

        dest = catalog.file_path(m)
        _fetch_all([(hf_hub_url(m.repo, m.filename), dest)], emit, m.size)
        return {"path": str(dest)}
    if m.kind == "separator":
        return _separator(m, emit)
    return _llama(payload.get("driver_cuda") or 0.0, emit)


def fetch(url: str, dest: Path, report: Callable[[int, int], None] | None = None) -> None:
    """下載到 dest.part，完成才改名成 dest。已經有 .part 時用 Range 接著下載。"""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    timeout = httpx.Timeout(30.0, read=120.0)
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=timeout) as r:
        if r.status_code == 416:  # 已經下載完，只是還沒改名
            os.replace(part, dest)
            return
        r.raise_for_status()
        if r.status_code == 206:
            total = int(r.headers.get("content-range", "/0").rsplit("/", 1)[-1] or 0)
            mode = "ab"
        else:  # 伺服器不支援續傳：從頭來
            total = int(r.headers.get("content-length") or 0)
            have, mode = 0, "wb"
        with open(part, mode) as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                have += len(chunk)
                if report:
                    report(have, total)
    os.replace(part, dest)


def _fetch_all(items: list[tuple[str, Path]], emit, total: int) -> None:
    progress = _Progress(emit, total)
    done_before = 0
    for url, dest in items:
        if dest.exists():
            done_before += dest.stat().st_size
            continue
        fetch(url, dest, lambda have, _t: progress(done_before + have, dest.name))
        done_before += dest.stat().st_size
    progress(done_before, force=True)


def _hf_repo(m: ModelInfo, emit) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    from huggingface_hub.file_download import are_symlinks_supported

    info = HfApi().model_info(m.repo, files_metadata=True)
    total = sum(s.size or 0 for s in info.siblings) or m.size
    progress = _Progress(emit, total)
    result: list = []
    # snapshot_download 用多個執行緒同時下載，各自檢查「這個資料夾能不能建符號連結」；檢查還沒做完時
    # 其他執行緒會先當成可以，在沒開開發人員模式的 Windows 上就會失敗（WinError 1314）。先檢查好再下載。
    root = catalog.repo_dir(m.repo)
    root.mkdir(parents=True, exist_ok=True)
    are_symlinks_supported(str(root))

    def run() -> None:
        try:
            result.append(snapshot_download(m.repo))
        except BaseException as e:  # noqa: BLE001 — 交給主執行緒丟出
            result.append(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    while thread.is_alive():  # snapshot_download 沒有逐位元組的回呼，改成看資料夾長多大
        progress(min(catalog.tree_size(root), total), m.repo)
        thread.join(0.5)
    if isinstance(result[0], BaseException):
        raise result[0]
    progress(total, force=True)
    return {"path": str(result[0])}


def _separator(m: ModelInfo, emit) -> dict:
    import httpx

    dest = catalog.file_path(m)
    config = dest.with_suffix(".yaml")
    items = []
    for target in (dest, config):
        for prefix in m.urls:  # 跟 audio-separator 一樣：先找 UVR 的 repo，找不到再找 audio-separator 的
            url = f"{prefix}/{target.name}"
            if httpx.head(url, follow_redirects=True, timeout=30).status_code == 200:
                items.append((url, target))
                break
        else:
            raise RuntimeError(f"找不到 {target.name} 的下載位置")
    _fetch_all(items, emit, m.size)
    return {"path": str(dest)}


def pick_llama_assets(releases: list[dict], driver_cuda: float) -> tuple[str, list[dict]]:
    """從 GitHub 發行版清單挑出最新一版適合這台電腦的 Windows 版本。

    有 NVIDIA 驅動時挑驅動支援得了的最新 CUDA 版本（另外要一起下載 cudart），否則用 CPU 版。
    回傳（版本, [asset, ...]）。
    """
    for rel in releases:
        assets = {a["name"]: a for a in rel.get("assets", [])}
        tag = rel.get("tag_name", "")
        cudas = []
        for name in assets:
            match = re.fullmatch(rf"llama-{re.escape(tag)}-bin-win-cuda-(\d+)\.(\d+)-x64\.zip", name)
            if match:
                version = float(f"{match.group(1)}.{match.group(2)}")
                if version <= driver_cuda:
                    cudas.append((version, name))
        if cudas:
            version, name = max(cudas)
            runtime = next((n for n in assets if re.fullmatch(rf"cudart-llama-bin-win-cuda-{version:g}-x64\.zip", n)),
                           None)
            if runtime:
                return tag, [assets[name], assets[runtime]]
        cpu = f"llama-{tag}-bin-win-cpu-x64.zip"
        if driver_cuda <= 0 and cpu in assets:
            return tag, [assets[cpu]]
    raise RuntimeError("在 llama.cpp 最近的發行版裡找不到適合的 Windows 版本")


def _llama(driver_cuda: float, emit) -> dict:
    import httpx

    releases = httpx.get(RELEASES_API, follow_redirects=True, timeout=30,
                         headers={"Accept": "application/vnd.github+json"}).raise_for_status().json()
    tag, assets = pick_llama_assets(releases, driver_cuda)
    root = catalog.llama_root()
    target = root / tag
    if (target / "llama-server.exe").exists():
        return {"path": str(target / "llama-server.exe"), "version": tag}
    staging = root / "_download"
    zips = [(a["browser_download_url"], staging / a["name"]) for a in assets]
    _fetch_all(zips, emit, sum(a["size"] for a in assets))

    emit({"type": "progress", "done": 0, "total": 0, "text": "解壓縮中…"})
    partial = root / f"{tag}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    for _, z in zips:
        with zipfile.ZipFile(z) as archive:
            archive.extractall(partial)
    exe = next(partial.rglob("llama-server.exe"), None)
    if exe is None:
        raise RuntimeError("下載的壓縮檔裡沒有 llama-server.exe")
    if exe.parent != partial:  # 有些版本的壓縮檔多包了一層資料夾
        for item in exe.parent.iterdir():
            shutil.move(str(item), partial / item.name)
    os.replace(partial, target)
    shutil.rmtree(staging, ignore_errors=True)
    return {"path": str(target / "llama-server.exe"), "version": tag}
