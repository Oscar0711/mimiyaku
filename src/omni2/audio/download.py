"""用 yt-dlp 下載影片網址的音訊。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def download_audio(url: str, dest_dir: Path, progress: Callable[[float], None] | None = None) -> tuple[Path, str]:
    """下載最佳音質的音軌到 dest_dir/source.<副檔名>，回傳（路徑, 標題）。"""
    import yt_dlp

    def hook(d: dict) -> None:
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress(min(1.0, d.get("downloaded_bytes", 0) / total))

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    return path, info.get("title") or path.stem
