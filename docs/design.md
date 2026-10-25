# Mimiyaku（原名 OmniTranslate v2）：重新設計計畫

> 2026-10-25 公開前改名為 Mimiyaku（耳訳），因為 OmniTranslate 和其他軟體撞名。以下的開發紀錄保留當時的名稱。程式內部的套件名稱 `omni2`、舊的資料資料夾 `OmniTranslate2`（已經存在時繼續沿用）和認證管理員裡的舊金鑰（讀到時搬到新名稱底下）都還相容。

## Context
上一版（舊的「全能翻譯」）功能齊全，但架構很亂，使用體驗也不好。例如 `gui/control_panel.py` 一個檔案就有 76KB。llama-cpp-python 在 RTX 50 系列上要自己找 CUDA 13 的 wheel 才能跑。卸載模型時 VRAM 也不一定會釋放乾淨。
這次從零重寫，保留三個核心：即時翻譯、音檔／影片翻譯、可選的 STT 引擎（本地＋雲端）。翻譯只用本地模型。

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

### M2：檔案工作區介面（2026-10-25）
- 用 PySide6 6.12 寫成，入口是 `omni2-gui`（`src/omni2/ui/`）。
  - `workspace.py`：佇列、選項和進度。
  - `player.py`：播放器和字幕同步。
  - `terms_dialog.py`：術語確認對話框。
  - `bridge.py`：在背景執行緒跑 FileRunner，事件透過 Qt signal 傳回介面。
- **術語確認**：背景執行緒送出請求後，會停下來等介面回覆；按「停止」也能中斷等待。
- **選項**：會存進設定檔的 `file_mode`。每一列記住開始處理時用的設定，處理完再改選項，預覽仍然找得到結果。
- **用 pythonw 啟動時**，`sys.stdout`、`sys.stderr` 是 `None`。主程式和每個 worker 子行程的入口都會先換成 devnull，否則 tqdm 進度條一寫就會當掉。
- **驗證**：
  - 用 offscreen 模式實際跑過兩個檔案（翻譯＋術語先行＋講者分離），共 26 秒；術語對話框和播放器同步都正常。
  - 在正常的 Windows 平台上，QMediaPlayer 讀得到長度，跳轉和播放也正常。
  - 注意：用 multiprocessing（spawn）的腳本一定要有 `if __name__ == "__main__":`，否則子行程會重跑整支腳本。

### M2：講者分離改進、pyannote、儲存位置（2026-10-25）
- **使用者回報：真實錄音大多被分成同一人。** 舊做法是一整句算一個聲紋。STT 為了準確會把短語音（不到 3 秒）併進下一段，所以「はい」「見た見た」這類短回應常和別人的話黏在同一句，聲紋被稀釋。
- **新做法**（`post/diarize.py`）：
  - 在每句範圍內滑動 1.5 秒的窗口，每 0.75 秒一個，各算一個 ECAPA 聲紋。
  - 用平均連結的階層式聚類，門檻 0.7，再把佔比不到 5% 的小群併回最像的大群。
  - 逐詞判斷講者，不到 0.6 秒的換人視為誤判。
  - 在換人的地方把句子切開。講者分離因此會產生新的分段（`diarized.json`），術語、翻譯、輸出都改用它。
- **評測**（`tools/bench_diarize.py`）：「在有語音的時間裡標對講者的比例」，講者名稱用匈牙利演算法做最佳對應。

  | 樣本 | 舊做法 | 新做法（自動） | 新做法（指定人數） |
  |---|---|---|---|
  | chat_ja（3 位女性快速聊天、很多短回應） | 69%（2 人） | 96%（3 人） | 96% |
  | meeting_ja（2 人） | 100% | 100% | 100% |
  | podcast_en（2 男 1 女、停頓短） | 71% | 98%（3 人） | 98% |
  | 4 段單人樣本 | 100% | 帶背景音樂那段 97%（多分 1 人），其餘 100% | 100% |

- **試過但不採用的做法**：
  - **用輪廓係數自動判斷人數**：單人錄音裡大喊的幾個窗口會自成一群，係數反而比真正的多人還高（0.41 對 0.19），把單人誤分成兩人。
  - **指定人數時從聚類樹頂端湊人數**：湊到的第 N 群常常是一小撮離群窗口。
  - **指定人數時從聚類樹底端湊人數**：會把很多碎群硬併在一起。

  以上兩種湊人數的做法都只有 60～80%。最後改成以門檻 0.7 為起點：人數太多就合併質心最像的兩群；人數太少就往下切，直到夠為止。
- **門檻來自合成語音**：真人錄音可能需要調整。設定頁提供三段「區分靈敏度」（距離 0.6／0.7／0.8）。
- **pyannote**（`speaker-diarization-community-1`）是可選的引擎。token 由 `hf auth login` 存在本機，程式不經手。直接傳波形給它，不用 torchcodec 解碼。沒有登入時會顯示中文的設定說明。
  - 因為還沒有 token，pyannote 只驗證過 import 和錯誤訊息，**實際的分離效果還沒測**。
- **內建術語表修正**：
  - 「配信」補上說明，避免歌曲的「配信されてない」被翻成「還沒開始直播」。
  - 新增「アクスタ → 壓克力立牌」「グッズ → 周邊」（原本「アクスタ」被翻成「徽章」）。
- **儲存位置**（`core/storage.py`、設定分頁）：
  - 工作檔、模型、llama.cpp 可以搬到任何資料夾；可以選擇把整個 Hugging Face 快取一起搬過去。
  - 同一顆磁碟直接改名；跨磁碟逐檔複製並回報進度，全部複製完才刪除原本的檔案。設定裡指向舊位置的路徑會自動改寫。
  - 資料夾位置透過環境變數 `OMNI2_DATA_DIR`、`HF_HOME` 傳給子行程。使用者原本的 `HF_HOME` 會記在 `OMNI2_DEFAULT_HF_HOME`，切回時用來還原。
  - 設定檔和術語表固定放在 `%LOCALAPPDATA%\OmniTranslate2`。

### M2：快速接話與重疊（2026-10-25）
- **使用者回報**：兩人話接得很近，或講話重疊時，講者會分錯。為了重現，新增了兩段用時間軸混音的樣本（`tools/make_samples.py` 的 `TIMELINE_SAMPLES`）：
  - `quick_ja`：三人快速接話，句間 0.05～0.15 秒。
  - `overlap_ja`：附和或插話時重疊 0.3～0.6 秒。
- **找到的原因**：
  1. 每人一次只講約 1 秒，VAD 又把整段對話併成一段，1.5 秒的窗口幾乎都橫跨兩個人，分群一開始就錯（多分出一人）。
  2. 不到 0.6 秒的換人被當成雜訊併掉，所以短回應會被吞掉。
  3. 窗口的時間解析度不夠，切點常差一兩個詞。
- **改進**（`post/diarize.py`）：
  - 先依句尾標點、逗號和 0.3 秒以上的停頓把文字切成「一句一句」，窗口不跨過這些邊界。
  - 兩階段：
    1. 長窗口分群，只用前後鄰居也是同一人的「乾淨」窗口算出每位講者的聲紋中心。
    2. 用 0.75 秒的短窗口、每 0.25 秒一個，逐段比對最像誰（分數會和相鄰窗口平均）。比對完再併一次小群，避免雜訊被放大成一位講者。
  - 短於 0.3 秒的換人才視為誤判（原本是 0.6 秒）。
  - 換人的切點往前後 2 個詞內「停頓最長或句尾」的位置靠。

  | 樣本 | 上一版 | 改進後 |
  |---|---|---|
  | quick_ja | 67%（4 人） | **93%（3 人）** |
  | overlap_ja | 86% | **92%** |
  | chat_ja | 96% | 98% |
  | podcast_en | 98% | 100% |
  | 4 段單人樣本 | 有一段多分 1 人 | 全部 100% |

- **剩下的錯誤大多來自語音辨識，講者分離修不了**：
  - 兩人的話被辨識成一個詞（「もう？」＋「うん」→「もううん」）。
  - 不到 0.3 秒的短詞（「え、」）沒有窗口，只能跟著前一句。
  - 重疊時，比較小聲、比較短的那句（「へえ！」「うんうん」）常常**根本沒被辨識出來**。文字裡不存在的話，就沒有辦法標講者。
  - 這類內容如果很多，pyannote 會更好：它會偵測重疊，並把重疊段落歸給主要講者。
- **試過但沒有效果**：把切句的停頓門檻從 0.3 秒降到 0.2 秒。

### M3：雲端語音辨識（2026-10-25）
- **金鑰**（`core/secrets.py`）：存在 Windows 認證管理員（keyring），沒存就讀環境變數 `DASHSCOPE_API_KEY`、`OPENAI_API_KEY`、`GOOGLE_APPLICATION_CREDENTIALS`。
  - 只在 STT 子行程裡讀取，不經過設定檔、工作參數或記錄檔。
  - Google 存的是服務帳戶 JSON 的**路徑**：認證管理員單筆最多約 2.5 KB，放不下整個 JSON。
- **Fun-ASR**（`stt/funasr_engine.py`）：
  - 音訊轉成 Opus（一小時約 15 MB）。
  - 用 `OssUtils.upload` 傳到 DashScope 暫存空間（`oss://`），多個檔案共用同一張上傳憑證、送成同一個工作。
  - 等工作完成後下載結果 JSON。句子和逐詞時間用 `stt/words.attach_punctuation` 把標點接回詞上。
  - 不回傳語言，所以用 `lang.guess_from_text` 依文字判斷。
  - 國際版（dashscope-intl）和中國版的金鑰不能互通。設定頁的「測試」用取得上傳憑證這個不收費的請求確認金鑰，目前的區域不通會自動試另一區。
- **OpenAI**（`stt/openai_engine.py`）和 **Google Speech v2 / Chirp 3**（`stt/google_engine.py`）：
  - 沿用本地的 VAD 切段，每段各送一次，最多 4 段同時進行（`stt/cloud_clips.py`）。
  - whisper-1 和 Google 有逐詞時間；gpt-4o-transcribe 只有文字，用 `spread_words` 依字寬估計時間。
  - **這兩個只用模擬的回應測過，還沒有真實金鑰可以驗證。**
- **介面**：檔案工作區有「辨識引擎」選單；設定分頁有各服務的金鑰欄位（密碼模式、存完即清空）、儲存／刪除／測試。

#### Fun-ASR 實測（國際版，2026-10-25）
- 用日、英、韓三段樣本（共約 3 分鐘）實測：

  | 引擎 | CER | 辨識時間 | 載入時間 |
  |---|---|---|---|
  | Fun-ASR（雲端） | 0.9% | 11.3 秒（含上傳） | 0.9 秒 |
  | Qwen3-ASR（本地） | 1.3% | 5.2 秒 | 25.8 秒 |

  - `fun-asr` 這個模型日、英、韓都支援。
  - 斷句比 Qwen3-ASR 細，一句一段，比較適合直接當字幕。
- **實測抓到的 bug**：回傳結果的 `file_url` 不是送出時的 `oss://` 網址，順序也不一樣，按順序對應時三個檔案的結果全部錯開。改成每個檔案用獨一無二的檔名（uuid），再從回傳的網址裡找出檔名來對應。
- **輸出的小毛病**，由 `tidy_text` 修正：
  - 日文會在停頓處插入空格：刪掉夾在兩個全形字之間的空格。
  - 英文句首和單獨的「i」常是小寫：改成大寫。
  - 韓文偶爾在詞中間多空格（「자 리가 많 지」）：韓文本來就用空格分詞，沒辦法用規則判斷，所以不處理。

### M4：即時翻譯（2026-10-25）
- **音訊來源**（`audio/sources.py`）：統一輸出 16 kHz 單聲道，每 0.1 秒一塊。
  - 麥克風、系統聲音：用 PyAudioWPatch（WASAPI loopback），先混成單聲道，再用 soxr 的串流重新取樣器轉成 16 kHz。
  - **指定程式**（`audio/process_loopback.py`）：用 comtypes 直接呼叫 `ActivateAudioInterfaceAsync("VAD\Process_Loopback")`，不需要編譯原生程式；要求系統直接轉成 16 kHz 單聲道。實測播放和擷取的音訊相關係數 1.000。
  - 注意：comtypes 被 import 的執行緒已經是 STA，`CoInitializeEx(MTA)` 會失敗。完成通知的處理器是 agile 物件，所以沿用 STA 也沒問題。
  - `FileSource`：以真實速度播放音檔，測試用。
- **切句**（`vad/stream.py`、`live/segmenter.py`）：
  - Silero v6 ONNX 串流執行，帶著 LSTM 狀態，每 32 毫秒輸出一次人聲機率。
  - 用兩個門檻做遲滯判斷；講話期間每 0.6 秒送出一次草稿；超過 8 秒在最近 2 秒內最像換氣的位置切開。
  - 預設情境是 conversation（停頓 0.5 秒才結束一句）。live 情境在逗號就會切開：日文 21 句、英文 25 句，參考答案只有 12、11 句，太碎了。
  - 往前補的緩衝要記住每一格真正的機率，否則很短的雜音會被當成一句話（單元測試抓到的 bug）。
- **辨識子行程**（`live/stt_process.py`、`live/recognizers.py`）：
  - 本地：切句器加上 `transcribe_text`（只要文字，Qwen3-ASR 不載入對齊器）。佇列積壓時跳過草稿，確定的句子一定會辨識。
  - 雲端：Fun-ASR 串流（`fun-asr-realtime`），原生就有草稿和確定兩種結果。
  - 以日文直播樣本、4 倍速餵入實測：

    | 引擎 | 準確度 | 每句辨識 | 顯示記憶體 | 啟動 |
    |---|---|---|---|---|
    | Qwen3-ASR 0.6B | 15 句都正確、標點完整 | 平均 0.56 秒 | 約 2.0 GB | 30 秒 |
    | faster-whisper turbo | 部分句子缺標點 | 平均 0.86 秒（beam 5；即時模式已改成 greedy） | 約 2.3 GB | 13 秒 |
    | Fun-ASR 即時 | 15 句都正確，草稿 100 次 | 雲端 | 0 | 1.7 秒 |

- **即時翻譯**（`live/translator.py`）：
  - 用 llama-server 串流逐字輸出純文字（不用 JSON）。帶前 3 句的原文和譯文、術語表。
  - 有快取；同語言跳過；模型多寫說明時只取第一行。
- **工作階段**（`live/session.py`）：
  - 辨識子行程和翻譯模型同時準備，翻譯模型先暖機，都好了才開始收音。
  - 停止時先停音源、等讀取執行緒結束，再收子行程最後的結果並翻完。子行程的結束訊號叫 `eof`，避免和工作階段的 `stopped` 撞名。
  - 紀錄寫在 `資料資料夾/sessions/*.jsonl`，可以匯出 SRT、TXT。多個來源時標上「我」「對方」。
- **端對端實測**：用 `FileSource` 以真實速度播放，Qwen3-ASR 0.6B＋Hy-MT2-7B：

  | 樣本 | 句數 | 說完→原文 | 說完→第一個譯字 |
  |---|---|---|---|
  | 日文直播 66 秒 | 15 句 | 平均 1.24 秒 | 平均 1.32 秒 |
  | 英文演講 56 秒 | 11 句 | 平均 1.66 秒 | 平均 1.73 秒 |

  「說完→原文」裡約有 0.5 秒是在等確認句子結束；講話期間，草稿每 0.6 秒更新一次。
- **介面**：
  - 懸浮字幕（`ui/overlay.py`）：無邊框、置頂、不搶焦點、半透明背景、文字加陰影；可拖曳、可調整大小；會淡出；可以設定滑鼠穿透。被擠掉的舊句子要先隱藏，否則會在左上角留下殘影。
    - 顯示幾句不固定：保留最近 30 句，從最新一句往回算每句在目前寬度下（含自動換行）需要的高度，放得下幾句就顯示幾句；字體變大，句數自動變少。
    - 版面配置要設成 `SetNoConstraint`，否則 Qt 會強制視窗至少容納所有內容，字幕一多視窗就縮不回去。
  - 全域快捷鍵（`ui/hotkeys.py`，RegisterHotKey）：Ctrl+Alt+S 顯示／隱藏、Ctrl+Alt+C 切換滑鼠穿透。
  - 即時分頁（`ui/live_page.py`）。即時和檔案模式互相停用「開始」，避免顯示記憶體不夠。
- **實機測試**：使用者用真的音源在 GUI 裡跑過完整的即時管線（2026-10-25），沒有大問題。

### M5：收尾（2026-10-25）
- **硬體偵測**（`core/hardware.py`）：只用 nvidia-smi 和 Windows API，不 import torch，開啟介面時不會卡。
  - 顯示卡型號、顯示記憶體、compute capability。
  - 驅動支援的 CUDA 版本：從 nvidia-smi 表頭讀。新版驅動寫成「CUDA UMD Version」。
  - CPU 名稱（登錄檔）、記憶體、資料資料夾所在磁碟的剩餘空間。
- **推薦**（`models/recommend.py`）：
  - 檔案翻譯每個階段輪流用顯示卡，所以挑單一模型放得下的最好版本。即時翻譯的兩個模型要同時放得下，必要時翻譯改用較小的模型。為此設定多了 `live.translate_model_path`，空字串＝和檔案翻譯相同。

    | 顯示記憶體 | 檔案翻譯 | 即時翻譯 |
    |---|---|---|
    | 12 GB 以上 | Qwen3-ASR 1.7B＋Hy-MT2 7B Q6_K | Qwen3-ASR 0.6B＋7B Q6_K（約 9 GB） |
    | 10 GB | 同上 | 0.6B＋7B Q4_K_M |
    | 8 GB | 1.7B＋7B Q4_K_M | 0.6B＋1.8B Q8_0 |
    | 6 GB | 0.6B＋1.8B Q8_0 | 0.6B＋1.8B Q8_0 |
    | 4 GB | Whisper turbo（int8）＋1.8B Q4_K_M | Whisper small＋1.8B Q4_K_M |
    | 沒有 NVIDIA | Whisper small（CPU int8）＋1.8B Q4_K_M（CPU） | 同左，建議改用 Fun-ASR |

  - 驅動只支援到 CUDA 12，或顯示卡比 Turing 舊時，PyTorch（cu130）不能用，所以 Qwen3-ASR、講者分離、人聲分離都不能用。這時改推 faster-whisper（CTranslate2 用 CUDA 12），llama.cpp 改下載 CUDA 12.4 版，並提醒更新驅動。
  - 使用者選了雲端辨識引擎時，套用推薦不會改掉。
  - 顯示記憶體：0.6B、7B Q6_K、1.8B Q4_K_M 是實測值，其餘依檔案大小估計。
  - Hy-MT2 1.8B Q4_K_M 用同一份評測句子實測（generic 風格）：

    | 模型 | chrF 日 | 英 | 韓 | 平均 | 秒／100 句 | VRAM |
    |---|---|---|---|---|---|---|
    | Hy-MT2-7B Q6_K（對照） | 42.3 | 52.1 | 50.5 | 48.3 | 37 | 7.1 GB |
    | Hy-MT2-1.8B Q4_K_M | 31.8 | 45.5 | 44.9 | 40.7 | 9.9 | 1.8 GB |

    1.8B 比 7B 差一截，但比舊的 Hunyuan-MT-7B（36.1）好，當小顯示卡和 CPU 的選項可以接受。Q8_0 沒有測。
- **模型目錄與下載**（`models/catalog.py`、`models/download.py`）：
  - 四種來源：HF repo（放在 HF 快取）、HF 單一檔案（GGUF，放在 models）、audio-separator 模型（GitHub）、llama.cpp（GitHub 發行版）。
  - 判斷是否下載完成：
    - HF 快取：沒有 `*.incomplete`；分片的權重要每一份都在。
    - 單一檔案：先下載到 `.part`，完成才改名。
  - 下載在子行程裡跑（`run_stage`），取消就直接結束子行程。
    - `.part` 用 Range 接著下載；HF 的 `*.incomplete` 由 huggingface_hub 自己續傳。
    - 實測：GGUF 取消兩次後，都從斷點接著下載。
  - HF repo 的進度：`snapshot_download` 沒有逐位元組的回呼，改成每 0.5 秒量一次資料夾大小。
  - **huggingface_hub 的競態**：`snapshot_download` 用多個執行緒同時下載。每個執行緒都會檢查資料夾能不能建符號連結，檢查做完之前，其他執行緒會先當成可以。在沒開開發人員模式的 Windows 上，這會在全新的快取資料夾觸發 WinError 1314。解法是下載前先呼叫一次 `are_symlinks_supported(repo 資料夾)`。
  - llama.cpp 的發行版現在都標成 prerelease，`releases/latest` 抓不到。改成列出最近 10 版，挑第一個有 Windows 版本的：
    - 有 NVIDIA 驅動時，挑驅動支援得了的最新 CUDA 版，加上對應的 cudart。
    - 沒有的話用 CPU 版。
    - 版本資料夾依 build 號排序（b11853），不看修改時間。
  - Silero VAD 包在 faster-whisper 裡，pyannote 要同意條款，這兩個不放進下載清單。
- **介面**：
  - 設定分頁改成左邊目錄、右邊一次一頁：硬體與模型、辨識與翻譯、雲端語音辨識、講者分離、儲存位置、其他。
  - 硬體與模型（`ui/models_page.py`）：
    - 顯示硬體和推薦。「套用推薦設定」會先列出要改的項目和要下載的模型，確認後才套用。
    - 模型表格：粗體＝目前設定會用到，每列都有下載／刪除按鈕。llama.cpp 可以更新、清除舊版。
    - 下載排成佇列一次一個，顯示速度和剩餘時間，可以個別取消。
    - 檔案或即時翻譯在跑的時候，不能刪除模型、不能套用推薦。
  - 辨識與翻譯：
    - 檔案翻譯和即時翻譯的模型可以分開選：Qwen3-ASR、Whisper、翻譯模型。翻譯模型清單會標出沒下載的；也可以選其他 GGUF。
    - 字幕排版：每行字數、行數、閱讀速度。
  - 其他：建立桌面／開始功能表捷徑、還原預設設定、開啟設定檔資料夾。
    - 還原預設設定會保留儲存位置、雲端設定、llama.cpp 位置和懸浮字幕位置，再依硬體套用推薦，然後重新開啟程式。
    - 設定是就地修改的，因為其他分頁拿著的是同一個物件。
  - 第一次開啟（`first_run_done`）：偵測硬體並跳出精靈（`ui/setup_dialog.py`），勾選要下載的模型。設定已經是推薦組合、模型也都在時，就不跳出。
  - 開始翻譯前先檢查 llama.cpp 和翻譯模型在不在。沒有的話提示到設定下載，不然 llama-server 會啟動失敗。辨識模型不檢查，因為第一次使用時會自動下載。
  - 表格換掉的按鈕要等事件迴圈才會刪除，期間會畫在錯的位置，所以換之前先藏起來。這和懸浮字幕的殘影是同一個問題。
- **捷徑與圖示**（`core/shortcut.py`、`tools/make_icon.py`）：
  - 捷徑指向 `.venv\Scripts\omni2-gui.exe`（uv 的 gui-scripts，不開主控台），用 WScript.Shell（comtypes）建立。
  - 圖示是程式產生的多尺寸 ICO，存在 `src/omni2/data/omni2.ico`。offscreen 平台不會自己讀 `.ttc`，要明確載入微軟正黑體。
  - 設定 AppUserModelID，工作列才會顯示我們的圖示，而不是 Python 的。
- **CLI**：
  - `omni2 doctor` 多了推薦組合。
  - 新增 `omni2 models list|recommend [--apply]|download <id…>` 和 `omni2 shortcut desktop start`。

