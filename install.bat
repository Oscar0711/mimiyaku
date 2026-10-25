<# : Mimiyaku installer (batch + PowerShell hybrid; the batch part only starts PowerShell)
@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -Command "$script = Get-Content -LiteralPath '%~f0' -Raw -Encoding UTF8; Invoke-Expression $script"
set code=%errorlevel%
if not defined OMNI2_NO_PAUSE pause
exit /b %code%
#>

# Mimiyaku 一鍵安裝／更新
#
# 第一次執行：安裝需要的工具（uv、Git、ffmpeg）、下載程式、安裝套件、建立桌面和開始功能表捷徑。
# 已經安裝過：再執行一次就會更新到最新版。
#
# 測試用的環境變數：OMNI2_INSTALL_DIR（安裝位置，不詢問）、OMNI2_REPO（下載來源）、
# OMNI2_NO_SHORTCUT（不建立捷徑）、OMNI2_NO_LAUNCH（裝完不開啟）、OMNI2_NO_PAUSE（結束時不暫停）

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Repo = if ($env:OMNI2_REPO) { $env:OMNI2_REPO } else { "https://github.com/Oscar0711/mimiyaku.git" }
$DefaultDir = Join-Path $env:LOCALAPPDATA "Programs\Mimiyaku"

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Info($text) { Write-Host "    $text" }
function Warn($text) { Write-Host "    ! $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host ""; Write-Host "安裝失敗：$text" -ForegroundColor Red; exit 1 }

function Refresh-Path {
    # winget 裝好的工具要重新讀 PATH 才找得到
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + `
                [Environment]::GetEnvironmentVariable("Path", "User")
}

function Has($command) { return [bool](Get-Command $command -ErrorAction SilentlyContinue) }

function Ensure-Tool($command, $wingetId, $label) {
    if (Has $command) { Info "$label 已安裝"; return }
    if (-not (Has "winget")) {
        Fail "找不到 winget。請先到 Microsoft Store 安裝「應用程式安裝程式」（App Installer），再執行一次。"
    }
    Info "安裝 $label…"
    & winget install --id $wingetId --exact --source winget
    Refresh-Path
    if (-not (Has $command)) { Fail "$label 安裝後還是找不到。請關掉這個視窗，再執行一次。" }
}

Write-Host "Mimiyaku 一鍵安裝" -ForegroundColor Green
Write-Host "會安裝 uv、Git、ffmpeg（已經有的會跳過）、下載程式並安裝套件。第一次安裝要下載數 GB，請耐心等候。"

if (-not [Environment]::Is64BitOperatingSystem) { Fail "需要 64 位元的 Windows。" }

# ---- 安裝位置 ----
Step "安裝位置"
if ($env:OMNI2_INSTALL_DIR) {
    $Dir = $env:OMNI2_INSTALL_DIR
} else {
    $answer = Read-Host "    直接按 Enter 安裝到 $DefaultDir，或輸入其他資料夾"
    $Dir = if ($answer.Trim()) { $answer.Trim().Trim('"') } else { $DefaultDir }
}
try { $Dir = [IO.Path]::GetFullPath($Dir) } catch { Fail "「$Dir」不是有效的資料夾路徑。" }
if ($Dir -eq [IO.Path]::GetPathRoot($Dir)) { Fail "不能直接裝在磁碟根目錄（$Dir），請指定一個資料夾，例如 $($Dir)Mimiyaku" }
$Updating = Test-Path -LiteralPath (Join-Path $Dir ".git")
if (-not $Updating -and (Test-Path -LiteralPath $Dir) -and (Get-ChildItem -LiteralPath $Dir -Force | Select-Object -First 1)) {
    Fail "$Dir 裡面已經有其他檔案。請指定一個新的或空的資料夾（例如 $(Join-Path $Dir 'Mimiyaku')）。"
}
Info $(if ($Updating) { "已經安裝在 $Dir，這次會更新到最新版" } else { "安裝到 $Dir" })

if (-not $Updating) {
    $drive = (Split-Path -Qualifier $Dir)
    $free = (Get-PSDrive -Name $drive.TrimEnd(':')).Free / 1GB
    if ($free -lt 15) { Warn ("{0} 只剩 {1:N0} GB；套件約 6 GB，模型另外需要 10～20 GB。" -f $drive, $free) }
}

# ---- 顯示卡 ----
Step "檢查顯示卡"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpu = (& $smi.Source --query-gpu=name,memory.total --format=csv,noheader) | Select-Object -First 1
    $banner = (& $smi.Source) -join "`n"
    Info "$gpu"
    if ($banner -match "CUDA (UMD )?Version:\s*([\d.]+)") {
        if ([double]$Matches[2] -lt 13.0) {
            Warn "驅動程式只支援到 CUDA $($Matches[2])。要用 Qwen3-ASR、講者分離、人聲分離，請先更新 NVIDIA 驅動程式。"
        }
    }
} else {
    Warn "沒有偵測到 NVIDIA 顯示卡。程式可以用 CPU 執行，但很慢，建議搭配雲端語音辨識。"
}

# ---- 工具 ----
Step "安裝需要的工具"
Refresh-Path
Ensure-Tool "uv" "astral-sh.uv" "uv（Python 套件管理）"
Ensure-Tool "git" "Git.Git" "Git"
Ensure-Tool "ffmpeg" "Gyan.FFmpeg" "ffmpeg"

# ---- 程式 ----
if ($Updating) {
    Step "更新程式"
    $running = Get-Process -Name "mimiyaku-gui", "omni2-gui" -ErrorAction SilentlyContinue
    if ($running) { Fail "Mimiyaku 正在執行，請先關掉它再更新。" }
    & git -C $Dir pull --ff-only
    if ($LASTEXITCODE -ne 0) { Fail "更新失敗。如果你改過程式的檔案，請先還原（git -C `"$Dir`" checkout .）再執行一次。" }
} else {
    Step "下載程式"
    # 上層資料夾不存在時 git clone 會自己建；不能對磁碟根目錄（例如 D:\）呼叫 New-Item，會報「路徑格式不正確」
    & git clone $Repo $Dir
    if ($LASTEXITCODE -ne 0) { Fail "下載程式失敗，請確認網路連線。" }
}

Step $(if ($Updating) { "更新套件" } else { "安裝套件（第一次要下載 PyTorch 等數 GB 的套件）" })
& uv sync --project $Dir
if ($LASTEXITCODE -ne 0) { Fail "安裝套件失敗。請確認網路連線，再執行一次（已下載的部分不會重新下載）。" }

# ---- 捷徑 ----
if (-not $env:OMNI2_NO_SHORTCUT) {
    Step "建立捷徑"
    & uv run --project $Dir omni2 shortcut desktop start
    if ($LASTEXITCODE -ne 0) { Warn "建立捷徑失敗，之後可以在程式的「設定 → 其他」建立。" }
}

Write-Host ""
if ($Updating) {
    Write-Host "更新完成！" -ForegroundColor Green
} else {
    Write-Host "安裝完成！" -ForegroundColor Green
    Write-Host "第一次開啟時，程式會偵測顯示卡並推薦要下載的模型。"
}
Write-Host "之後要更新，再執行一次這個檔案就好。"

if (-not $env:OMNI2_NO_LAUNCH) {
    $gui = Join-Path $Dir ".venv\Scripts\mimiyaku-gui.exe"
    if (Test-Path $gui) { Start-Process -FilePath $gui -WorkingDirectory $Dir }
}
exit 0
