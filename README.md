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

## 介面

```bash
uv run omni2-gui
```

「檔案翻譯」分頁：把影音檔拖進視窗（或貼上網址），勾選選項後按「開始」。
- 佇列會顯示每個檔案目前的階段和進度；雙擊已完成的檔案會開啟輸出資料夾。
- 勾選「全文術語先行」時，翻譯前會跳出對話框讓你確認摘要和專有名詞。
- 點選已完成的檔案，下方的播放器會同步顯示原文和譯文；點字幕清單可以跳到那個時間點。
- 按「停止」後，已完成的部分會保留，下次按「開始」會從斷點繼續。

「設定」分頁：
- 可以把工作檔、模型和 llama.cpp 搬到其他資料夾，也可以選擇把語音辨識模型（Hugging Face 快取）一起搬過去。
- 可以選擇講者分離的引擎和區分靈敏度。
- 要用 pyannote 的話：先到 [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) 同意使用條款，再執行 `uv run hf auth login` 登入。

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
