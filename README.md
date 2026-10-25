# OmniTranslate v2

本地語音辨識與字幕翻譯工具（Windows）。設計說明見 [docs/design.md](docs/design.md)。

## 安裝

```bash
uv sync
```

另外需要 ffmpeg（放在 PATH 上）。

模型不用自己找：第一次開啟介面時會偵測顯示卡，推薦一組放得下的模型，勾選後就會下載（包括翻譯用的 llama.cpp）。之後也可以在「設定 → 硬體與模型」下載、刪除、更新。命令列版本：

```bash
uv run omni2 models recommend --apply   # 依硬體套用推薦設定
uv run omni2 models list                # 每個模型的下載狀態
uv run omni2 models download hy-mt2-7b-q6 llama-server
uv run omni2 shortcut desktop start     # 在桌面和開始功能表建立捷徑
```

## 介面

```bash
uv run omni2-gui
```

「檔案翻譯」分頁：把影音檔拖進視窗（或貼上網址），勾選選項後按「開始」。
- 佇列會顯示每個檔案目前的階段和進度；雙擊已完成的檔案會開啟輸出資料夾。
- 勾選「全文術語先行」時，翻譯前會跳出對話框讓你確認摘要和專有名詞。
- 點選已完成的檔案，下方的播放器會同步顯示原文和譯文；點字幕清單可以跳到那個時間點。
- 按「停止」後，已完成的部分會保留，下次按「開始」會從斷點繼續。

「即時翻譯」分頁：
- 音源可以選系統聲音、只收指定程式（例如瀏覽器，不會收到 Discord 或通知音）、麥克風，或麥克風＋系統聲音（線上會議）。
- 原文會邊聽邊出現（灰色草稿），句子結束後才翻譯，譯文逐字長出來。
- 懸浮字幕可以拖曳、調大小、調透明度、設定滑鼠穿透。快捷鍵：Ctrl+Alt+S 顯示／隱藏字幕，Ctrl+Alt+C 切換滑鼠穿透。
- 每次即時翻譯都會留下紀錄，可以匯出成 SRT 或 TXT。

「設定」分頁：
- 硬體與模型：顯示偵測到的顯示卡和推薦的模型組合；每個模型可以下載、刪除，下載中斷後下次會接著下載。
- 辨識與翻譯：檔案翻譯和即時翻譯可以用不同的模型（例如顯示記憶體不夠時，即時翻譯改用小一點的翻譯模型）；字幕每行字數、行數、閱讀速度。
- 其他：建立桌面或開始功能表的捷徑、還原預設設定。
- 可以把工作檔、模型和 llama.cpp 搬到其他資料夾，也可以選擇把語音辨識模型（Hugging Face 快取）一起搬過去。
- 可以選擇講者分離的引擎和區分靈敏度。
- 可以填入雲端語音辨識的金鑰（阿里雲 Fun-ASR、OpenAI、Google）。金鑰存在 Windows 認證管理員，不會寫進設定檔；按「測試」可以確認金鑰是否有效。
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
