# OmniTranslate v2：重新設計計畫

## Context
上一版（`D:\claude\全能翻譯`）功能齊全，但架構很亂，使用體驗也不好。例如 `gui/control_panel.py` 一個檔案就有 76KB。llama-cpp-python 在 RTX 50 系列上要自己找 CUDA 13 的 wheel 才能跑。卸載模型時 VRAM 也不一定會釋放乾淨。
這次在 `D:\claude\translator` 從零重寫，保留三個核心：即時翻譯、音檔／影片翻譯、可選的 STT 引擎（本地＋雲端）。翻譯只用本地模型。

**已確認的決定**
- 介面：PySide6
- 開發順序：檔案模式先
- 用途：自己用，用 uv 直接跑原始碼
- 語言：日／英／韓 → 繁中
- 即時模式新功能：兩段式字幕、指定程式擷取、麥克風＋系統音同時收、即時紀錄匯出
- 檔案模式新功能：全文術語先行、斷點續作、人聲分離，以及**純紀錄模式**（只辨識不翻譯，含講者分離、TXT／MD 逐字稿輸出）
- 不做字幕編輯器。保留播放器（同步顯示原文和譯文，點字幕清單可跳到該時間點）
- 硬體：RTX 5060 Ti 16GB（Blackwell sm_120）、Ryzen 7 7700、32GB RAM，已裝 CUDA 13.0

**舊版只沿用資料，不沿用程式結構**
- `omnitrans/stt/hallucination.py`：片語黑名單和三種訊號的門檻（附實測數字）
- `config/vad_presets.yaml`：遲滯門檻和各情境的參數
- `requirements.txt` 裡在 Blackwell 上驗證過的安裝備註（torch cu13x、faster-whisper 可直接跑）

## 架構

```
UI 行程 (PySide6)  ── 只負責介面和調度，不載入任何模型
  ├─ STT worker 行程      (multiprocessing spawn；載入 STT 和 Silero VAD；結束行程＝VRAM 完整釋放)
  ├─ llama-server 子行程  (llama.cpp 官方預編譯 CUDA 版，OpenAI 相容 API，串流＋json_schema)
  └─ 短命 worker          (人聲分離、講者分離；檔案模式中逐階段啟動，做完就結束)
```

- **翻譯**：由 `llama-server` 負責。用 httpx 呼叫 `/v1/chat/completions`，開啟 `stream` 和 `cache_prompt`，批次翻譯時用 `response_format: json_schema`。換模型就是重啟 server。設定上也允許改指向任何 OpenAI 相容的本地端點（Ollama、LM Studio）。
- **Worker RPC**：用 multiprocessing Queue，訊息是 dataclass，音訊用 16k mono float32 numpy。worker 崩潰（OOM）時，UI 顯示錯誤並可以重啟。
- **STT 介面**（`stt/base.py`）：
  - `transcribe_file(audio, opts) -> list[Segment]`：包含詞級時間戳、`no_speech_prob`、`compression_ratio`。支援多檔一起推論時（Qwen3-ASR）提供 `transcribe_batch()`。
  - `stream(chunks) -> Iterator[Partial | Final]`：給即時模式用。本地引擎用 LocalAgreement-2 包裝；雲端引擎直接接各家的串流 API。
  - 各引擎：`faster_whisper`、`qwen3_asr`（含 forced aligner）、`sensevoice`（即時低延遲候選）、`openai`、`google`、`funasr`。
- **翻譯是可選的階段**：關掉就是純紀錄模式。即時和檔案模式共用同一套開關。

### 套件結構（`src/omni2/`）
| 目錄 | 內容 |
|---|---|
| `core/` | 資料型別（Segment、Word、Job、SessionEvent）、語言碼、事件 |
| `config/` | pydantic 設定。自動存到 `%LOCALAPPDATA%\OmniTranslate2\config.yaml`。金鑰用 `keyring`（Windows 認證管理員），欄位空白時改讀環境變數 |
| `audio/` | ffmpeg 解碼、yt-dlp、soxr 重取樣、麥克風、WASAPI loopback、process loopback |
| `vad/` | Silero VAD 和分段器（遲滯門檻、情境預設、音樂模式的固定窗口） |
| `stt/` | 介面、各引擎 adapter、`streaming.py`（LocalAgreement）、`hallucination.py` |
| `translate/` | `llama_server.py`（啟動、健康檢查、換模型）、`client.py`、`prompts.py`、`batch.py`（JSON schema 對齊）、`glossary.py`、`cache.py`、OpenCC `s2twp` 繁體保險 |
| `post/` | 字幕排版（行寬、行數、最短顯示時間、每秒字數、防重疊、依詞時間戳拆長段）、講者分離、人聲分離 |
| `workers/` | 子行程 host 和 RPC |
| `jobs/` | 檔案模式的 job 狀態機、斷點續作、階段排程 |
| `live/` | 即時管線、session 紀錄 |
| `export/` | SRT、VTT、ASS（可選雙語）、TXT、MD（時間戳和講者可選） |
| `ui/` | 主視窗、檔案工作區、播放器、術語確認對話框、懸浮字幕、設定。每個畫面一個模組，不再做成一個巨型 panel |
| `cli.py` | 不開 UI 的檔案模式指令，方便測試和批次處理 |

## 檔案模式（先做）

**分階段流程**：每一階段結束都寫斷點；每一階段用獨立行程，做完就釋放 VRAM。
1. **取得音訊**：本地檔案或 yt-dlp 抓網址，再用 ffmpeg 轉成 16k mono wav，存進 job 資料夾。
2. **人聲分離**（可選）：用 `audio-separator`（Mel／BS-RoFormer vocals 模型）產生 `vocals.wav` 並快取。
3. **STT**：所有檔案一起辨識（Qwen3-ASR 用批次推論），經過幻覺過濾，結果寫入 `transcript.jsonl`。
4. **講者分離**（可選，純紀錄模式預設開啟）：用 pyannote `speaker-diarization-community-1`，依時間重疊把講者標到每個片段上。
5. **全文術語先行**（可選）：LLM 讀完全文（太長時分塊），用 JSON schema 輸出摘要和專有名詞表（原文、建議譯法、類型）。UI 跳出可編輯的表格，你確認後才繼續。確認過的條目可以選擇併入全域術語表。
6. **翻譯**：每 10 句送一批，用 JSON schema 強制每個 id 都有一句譯文，確保句數對齊。Prompt 包含全文摘要、這批句子用到的術語、前 K 句的原文＋譯文。快取重複出現的句子；來源語言和目標語言相同時跳過。翻完做輕量品質檢查（空白、照抄原文、語言錯誤），再加 OpenCC 繁體保險。進度寫入 `translation.jsonl`，每批寫一次。
7. **排版和匯出**：輸出 `原檔名.zh-TW.srt` 等檔案。純紀錄模式輸出 `原檔名.ja.srt`、`.txt` 和 `.md`。

**斷點續作**：每個 job 一個資料夾，放在 `%LOCALAPPDATA%\OmniTranslate2\jobs\<hash>\`，裡面有 `job.json`（各階段狀態和設定快照）、各階段產物和 jsonl。重開後從第一個沒完成的階段或批次接著做。只有完整做完的 job 才寫出最終字幕。

**工作區 UI**：批次佇列顯示每個檔案目前在哪個階段和進度，可以停止和續作。播放器同步顯示原文和譯文，點字幕清單可以跳到該時間點。

## 即時模式（後做）
- **擷取**：麥克風、系統 loopback、指定程式（process loopback）、麥克風＋系統音同時收。同時收時，每個音源各自做 VAD，共用同一個 STT worker，片段標上「我」或「對方」。
  - process loopback 先找現成的 Python 套件；沒有的話，用微軟 ApplicationLoopback 範例寫一個小型 C++ helper，把 PCM 從 stdout 傳回來（放在 `native/`）。**這是最大的技術風險。**
- **兩段式字幕**：串流 STT 持續送出 `Partial`，原文以灰色即時顯示。`Final` 送出後才翻譯，譯文逐字長出來。用 segment id 保證順序。VAD 偵測句尾，LocalAgreement 決定穩定前綴；長句在句中出現穩定的句讀時先提交，取代原本 6 秒強制切段的做法。
- **懸浮字幕**：無邊框、永遠置頂、半透明、可拖曳、沒人說話時淡出、字體大小和透明度可調、滑鼠穿透（`WS_EX_TRANSPARENT`），以及全域快捷鍵。
- **兩個模型同時載入**，載入後先暖機一次。
- **即時紀錄**：每次 session 寫成 JSONL，結束後可匯出 SRT 或 TXT。

## 模型（在 M0 和 M1 實測後定預設）
| 用途 | 候選 |
|---|---|
| 本地 STT | faster-whisper large-v3-turbo（預設）、Qwen3-ASR-1.7B＋aligner（檔案高精度、歌曲）、SenseVoice-Small（即時低延遲） |
| 翻譯 | Hunyuan-MT-7B、TranslateGemma-12B、Qwen3-8B（關閉 thinking），都用 GGUF。專門翻譯模型如果不擅長抽術語，術語階段改用通用模型（只需重啟 server） |
| 雲端 STT | OpenAI（gpt-4o-transcribe／Realtime）、Google Speech v2（Chirp）、阿里雲 Fun-ASR（檔案＋realtime） |
| 講者分離 | pyannote community-1 |
| 人聲分離 | audio-separator（RoFormer vocals） |

VRAM 預算：turbo 約 2GB，7B Q6 約 6.5GB，加上 KV cache 約 2GB，16GB 放得下。

## 里程碑
- **M0 環境驗證**：建立 uv 專案（Python 3.12）。在 Blackwell 上逐一確認 torch CUDA 版、faster-whisper、qwen-asr、llama-server CUDA 版、pyannote、audio-separator 都能跑。
- **M1 檔案模式核心（CLI）**：做 STT 介面＋faster-whisper、llama-server 翻譯（JSON 批次）、排版和匯出、job 斷點。用日、英、韓三個樣本實測比較翻譯模型。
- **M2 檔案模式完整**：加入 Qwen3-ASR 批次推論、人聲分離、講者分離、全文術語先行、純紀錄模式、yt-dlp，以及 PySide6 工作區和播放器。
- **M3 雲端 STT**：OpenAI、Google、Fun-ASR 的檔案版，金鑰分頁用 keyring。
- **M4 即時模式**：擷取（四種音源）、串流 STT 和兩段式字幕、懸浮字幕、即時紀錄、雲端串流 STT。
- **M5 收尾**：首次啟動偵測硬體並推薦模型組合、自動下載模型（顯示進度）、設定頁、啟動捷徑。

## 驗證
- **pytest 單元測試**：排版、SRT／VTT／ASS／TXT／MD 匯出、幻覺過濾、JSON 批次對齊（模擬模型漏句或多句）、斷點續作（在各階段中斷後重跑，結果應與一次跑完相同）、術語注入。
- **整合測試**：準備日、英、韓三段 1～2 分鐘的樣本，跑 `omni2 file sample.wav --to zh-TW`，檢查字幕句數、時間軸和繁體輸出。純紀錄模式加講者分離時，用兩人對話樣本檢查講者標記。
- **效能**：用 `tools/bench.py` 記錄各引擎的 RTF、VRAM 峰值，以及翻譯樣本的對照表。
- **即時模式**：在 UI 加一個除錯 HUD，顯示「說完到原文出現」和「說完到第一個譯字出現」的延遲。用 YouTube 加 process loopback 實測，確認 Discord 的聲音不會被收進來。

## 實作紀錄

### M0：環境（2026-10-24）
- llama.cpp b11853 `win-cuda-13.4-x64`（驅動 617.14 支援 CUDA 13.4），放在 `%LOCALAPPDATA%\OmniTranslate2\bin\llama.cpp\b11853\`。
- CTranslate2 4.8.3、faster-whisper 1.2.1 在 RTX 5060 Ti 上可用 float16。
- torch 尚未安裝；等到 M2 的 Qwen3-ASR、pyannote 和人聲分離才需要。
- 測試樣本用 `tools/make_samples.py`（edge-tts）產生，附每句的正確文字和時間。

### M1：檔案模式核心的發現
- **不用 faster-whisper 內建的 VAD 合併。** `BatchedInferencePipeline` 會把多個語音區段接起來、拿掉中間的靜音，造成兩個問題：
  - 日文輸出完全沒有標點。
  - 詞級時間戳在句子交界處錯位，下一句的第一個字會被歸到上一段。

  改成自己用 Silero 算區段，透過 `clip_timestamps` 讓每個區段各自解碼並保留停頓。日文樣本的 12 段邊界因此全部正確，65 秒音訊約 5 秒辨識完。
- **不用標點提示句（initial_prompt）。** 它雖然能補上標點，但換一句提示句後，曾經有約 20 秒的語音被整段跳過。
- 已知限制：Whisper 常省略句首的短感嘆詞（例如「はい」）。
- Qwen3-8B Q4_K_M 可以用，但偶爾會出現錯字（氣餒 → 氣餓）和專有名詞錯譯（성수동 → 成秀洞）。下一步要做翻譯模型比較。

### M1：翻譯模型比較（2026-10-24）
- 評測資料：`tools/bench_data/translate.json`，共 76 句（日 33、英 23、韓 19）。內容包括直播、會議、演講、口語、韓劇對白，每句附繁中參考譯文。
- 執行方式：`uv run python tools/bench_translate.py 名稱=GGUF[:風格[@溫度]] ...`，報告輸出到 `samples/bench/report.md`。

| 模型 | 風格 | chrF 日 | 英 | 韓 | 平均 | 殘留簡體（字） | 秒／100 句 | VRAM |
|---|---|---|---|---|---|---|---|---|
| **Hy-MT2-7B Q6_K** | generic | 42.3 | 52.1 | 50.5 | **48.3** | 0 | 37 | 7.1 GB |
| Qwen3-8B Q4_K_M | generic | 34.5 | 46.7 | 53.9 | 45.0 | 0 | 26 | 5.8 GB |
| Hy-MT2-7B Q6_K | hy-mt（官方格式） | 34.9 | 49.1 | 41.9 | 42.0 | 0 | 39 | 7.1 GB |
| Hunyuan-MT-7B Q6_K | generic | 31.6 | 39.2 | 37.6 | 36.1 | 4 | 38 | 7.2 GB |
| TranslateGemma-12B Q4_K_M | translategemma（官方，逐句） | 27.2 | 40.9 | 34.8 | 34.3 | 80 | 41 | 8.2 GB |

同一個模型重跑一次，分數大約會浮動 ±1.5，所以結論以逐句閱讀為準：
- **預設改用 Hy-MT2-7B。** Qwen3-8B 有好幾句嚴重誤譯，例如：
  - 아이고 깜짝이야 → 驚喜啊
  - スパチャ → 斯巴達感謝
  - 眠すぎる → 睡太多了
  - 성수동 → 成駅洞
  - 「ninety」沒有翻

  這些 Hy-MT2 都翻對了。
- **Hy-MT 系列用 generic 格式。** 官方格式就算溫度降到 0.2 也只有 40.6 分，譯文冗長，又到處加「哦、啦」。
- **TranslateGemma 不採用。** 意思大致正確，但很正式又冗長（請您慢慢享用、轉帳），大多輸出簡體字，也沒有放前文的位置。
- **繁體保險改成只轉換確定是簡體的字。** 無條件套 OpenCC 會把正確的繁體改壞，例如「濃郁」→「濃鬱」。
- **譯文後處理會去掉不成對的引號。** Hy-MT2 偶爾會在句尾多一個「」」。
- **共通弱點：** 直播用語（草、初見さん、歌枠）和專有名詞的譯名不穩定。同一個「성수동」會隨原文的小差異在「聖水洞」和「城西洞」之間浮動。這要靠 M2 的全文術語先行，再加一份內建的直播用語術語表來解決。

### M2：檔案模式完整（2026-10-25）
- **torch 2.11 + cu130**：torchaudio 在 CUDA 13 只出到 2.11，SpeechBrain 又需要它，所以鎖在這個版本；sm_120 實測可用。`pyproject.toml` 用 `tool.uv.environments` 限定只解 Windows 的相依。
- **Qwen3-ASR 設為預設 STT。** 用 4 段樣本（日 2、英、韓）比較：

  | 引擎 | CER | 標點 | 辨識時間 | 載入時間 |
  |---|---|---|---|---|
  | Qwen3-ASR-1.7B＋ForcedAligner | **1.3%** | 日、韓、英都完整 | 20.8 秒 | 22 秒 |
  | faster-whisper large-v3-turbo | 2.2% | 日文幾乎沒有 | 16.5 秒 | 3.7 秒 |

  - 兩者都會吃掉句首很短的「はい」。
  - Qwen3-ASR 會把所有檔案的所有語音區段放進同一次批次推論，每組上限 2 小時音訊。
  - 對齊器輸出的詞會去掉標點，所以由 `attach_punctuation` 依序把原文的標點接回每個詞。
- **人聲分離**：用 audio-separator 的 `vocals_mel_band_roformer`。從原始檔解出 44.1k 立體聲再分離，人聲轉成 16k 給 STT。模型約 0.9 GB，第一次下載很慢（GitHub，約 1 MB/s）。用合成背景音樂測試時，有沒有分離的 CER 都是 3.3%，合成音樂太好分辨，還沒能證明準確率有提升，要用真實影片驗證。
- **講者分離**：每句算一個 SpeechBrain ECAPA 聲紋，再做平均連結的階層式聚類（餘弦距離 0.65）。不到 0.8 秒的短句事後指派給最像的講者。
  - 不用 pyannote：它需要 Hugging Face 帳號同意條款，這台電腦沒有 token。
  - 合成的兩人會議 9 句全部分對，但兩個合成語音差異很大，真人錄音還沒驗證。
- **全文術語先行**：用翻譯模型 Hy-MT2 抽術語，不換模型。實測它比 Qwen3-8B 精準：Qwen3 會列出一般詞，譯名也錯（成數洞、スパチャ→刷金幣）。
  - 內建表和使用者術語表當作已知詞，連同譯名一起告訴模型。這樣摘要會照用固定譯名，不會再寫出「城수洞」，也不會重複列出已知詞。
  - 逐字稿裡沒出現的詞會被丟掉，避免模型自己編。
  - CLI 在互動式終端機裡會寫出 `terms.yaml` 讓使用者確認。Windows 的 NUL 裝置 `isatty()` 也回傳 True，所以要連輸出也是終端機才算互動模式。
- **內建術語表** `src/omni2/data/glossary/streaming.{ja,ko,en}.yaml`：直播用語和首爾地名，只用在翻譯，不當 STT 提示詞。英文詞整詞比對。優先順序：內建 < 使用者 < 這份檔案確認過的術語。
- **網址輸入**：用 yt-dlp 下載最佳音軌。用本機 HTTP 伺服器測試，下載、轉檔、辨識都可以；真實影片網站還沒測。
