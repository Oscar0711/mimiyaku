# OmniTranslate v2

本地語音辨識與字幕翻譯工具（Windows）。設計說明見 [docs/design.md](docs/design.md)。

## 安裝

```bash
uv sync
```

另外需要：
- ffmpeg（放在 PATH 上）
- llama.cpp 的 `llama-server`（Windows CUDA 版），路徑寫在設定檔的 `translate.server_exe`
- 一個翻譯用的 GGUF 模型，路徑寫在 `translate.model_path`。建議用 [Hy-MT2-7B](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) 的 Q6_K（比較結果見設計文件）

## 檔案翻譯（CLI）

```bash
uv run omni2 file 影片.mp4 --to zh-TW
uv run omni2 file 會議.m4a --no-translate --diarize --formats md,txt
uv run omni2 file 直播.mp4 --terms --bilingual        # 翻譯前先整理術語，確認後再翻
uv run omni2 file MV.mp4 --vocals                      # 有背景音樂時先分離人聲
uv run omni2 file https://www.youtube.com/watch?v=...  # 影片網址
uv run omni2 doctor
```

設定檔：`%LOCALAPPDATA%\OmniTranslate2\config.yaml`（第一次執行時自動建立）。
