"""命令列介面：不開 UI 也能跑檔案模式，方便測試和批次處理。"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from omni2.config.settings import Settings, load_settings
from omni2.core import paths
from omni2.export.formats import ALL_FORMATS
from omni2.translate.glossary import Glossary
from omni2.vad.presets import PRESETS


class ConsoleReporter:
    """把 runner 的事件印成一行一行的進度。"""

    def __init__(self) -> None:
        self.tty = sys.stdout.isatty()
        self._last: dict[tuple[str, str], int] = {}
        self._inline = False

    def _print(self, text: str) -> None:
        if self._inline:
            sys.stdout.write("\n")
            self._inline = False
        print(text, flush=True)

    def __call__(self, ev: dict) -> None:
        if ev["type"] == "status":
            self._print(f"» {ev['message']}")
            return
        if ev["type"] != "stage":
            return
        job, stage, status = ev["job"], ev["stage"], ev["status"]
        from omni2.jobs.runner import STAGE_LABELS

        label = STAGE_LABELS[stage]
        if status == "progress":
            pct = int(ev["value"] * 100)
            if self.tty:
                sys.stdout.write(f"\r  {job}｜{label} {pct:3d}%")
                sys.stdout.flush()
                self._inline = True
            elif pct // 10 > self._last.get((job, stage), -1):
                self._last[(job, stage)] = pct // 10
                print(f"  {job}｜{label} {pct}%", flush=True)
            return
        text = {"start": "開始", "done": "完成", "skip": "沿用斷點", "error": "失敗"}[status]
        message = ev.get("message")
        self._print(f"  {job}｜{label}{text}" + (f"：{message}" if message else ""))


def console_review(job, result):
    """在終端機裡確認抽出來的術語：寫成可編輯的 YAML，等使用者按 Enter。"""
    import yaml

    from omni2.core.jsonio import write_text_atomic
    from omni2.translate.terms import TermsResult

    path = job.dir / "terms.yaml"
    body = {"summary": result.summary,
            "terms": [{"source": t["source"], "target": t["target"], "note": t.get("note", "")}
                      for t in result.terms]}
    header = "# 確認後存檔：可以修改譯名、刪掉不要的條目，也可以自己加。summary 會提供給譯者當背景。\n"
    write_text_atomic(path, header + yaml.safe_dump(body, allow_unicode=True, sort_keys=False, width=1000))
    print(f"\n── {job.name} 的術語 ──")
    print(f"摘要：{result.summary}")
    for t in result.terms:
        note = f"（{t.get('note')}）" if t.get("note") else ""
        print(f"  {t['source']} → {t['target']}{note}")
    print(f"可以編輯 {path}，存檔後按 Enter 繼續；輸入 s 表示這份檔案不使用術語。")
    try:
        answer = input("> ")
    except EOFError:  # 沒有人能回答（例如被其他程式呼叫）：直接採用
        print("（沒有輸入，直接採用）")
        return result
    if answer.strip().lower() == "s":
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return TermsResult(summary=str(data.get("summary") or ""),
                       terms=[t for t in (data.get("terms") or []) if isinstance(t, dict) and t.get("source")])


def _setup_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("    · %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.WARNING)
    if verbose:
        for noisy in ("httpx", "httpcore", "faster_whisper"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_file(args: argparse.Namespace, settings: Settings) -> int:
    from omni2.jobs.job import JobSpec
    from omni2.jobs.runner import FileRunner

    if args.stt:
        settings.stt.engine = args.stt
    if args.stt_model:
        if settings.stt.engine == "qwen3-asr":
            settings.stt.qwen_model = args.stt_model
        else:
            settings.stt.model = args.stt_model
    if args.vad:
        settings.stt.vad_preset = args.vad
    if args.model:
        settings.translate.model_path = args.model
    formats = [f.strip() for f in args.formats.split(",")] if args.formats else settings.output.formats
    bad = [f for f in formats if f not in ALL_FORMATS]
    if bad:
        print(f"不支援的格式：{', '.join(bad)}（可用：{', '.join(ALL_FORMATS)}）")
        return 2

    glossary = Glossary.load(paths.glossary_path())
    if args.glossary:
        glossary = glossary.merged(Glossary.load(Path(args.glossary)))

    translate = settings.translate.enabled and not args.no_translate
    specs = []
    for item in args.inputs:
        if not item.startswith(("http://", "https://")) and not Path(item).exists():
            print(f"找不到檔案：{item}")
            return 2
        specs.append(JobSpec(
            input=item,
            src_lang=args.src,
            translate=translate,
            tgt_lang=args.tgt or settings.translate.target,
            formats=formats,
            bilingual=args.bilingual or settings.output.bilingual,
            out_dir=args.out,
            separate_vocals=args.vocals,
            diarize=args.diarize or bool(args.speakers),
            num_speakers=args.speakers,
            terms_first=args.terms,
        ))

    review = None
    # Windows 的 NUL 裝置 isatty() 也回傳 True，所以輸出也要是終端機才算互動模式
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if args.terms and settings.terms.review and not args.no_review and interactive:
        review = console_review
    runner = FileRunner(specs, settings, glossary=glossary, on_event=ConsoleReporter(), review=review,
                        jobs_root=Path(args.jobs_dir) if args.jobs_dir else None)
    try:
        jobs = runner.run()
    except KeyboardInterrupt:
        print("\n已停止。再執行一次同樣的指令，會從斷點繼續。")
        return 130
    failed = [j for j in jobs if j.error]
    print()
    for job in jobs:
        if job.error:
            print(f"✗ {job.name}：{job.error}")
        else:
            for p in job.outputs:
                print(f"✓ {p}")
    return 1 if failed else 0


def cmd_doctor(settings: Settings) -> int:
    ok = True

    def line(good: bool, label: str, detail: str) -> None:
        nonlocal ok
        ok &= good
        print(f"{'✓' if good else '✗'} {label}：{detail}")

    line(True, "Python", sys.version.split()[0])
    ffmpeg = shutil.which("ffmpeg")
    line(bool(ffmpeg), "ffmpeg", ffmpeg or "找不到，請安裝並加入 PATH")
    try:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        line(n > 0, "CTranslate2 CUDA", f"{ctranslate2.__version__}，{n} 張 GPU")
    except Exception as e:  # noqa: BLE001
        line(False, "CTranslate2", str(e))
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        line(True, "GPU", out)
    exe = Path(settings.translate.server_exe) if settings.translate.server_exe else None
    if settings.translate.endpoint:
        line(True, "翻譯端點", settings.translate.endpoint)
    else:
        line(bool(exe and exe.exists()), "llama-server", str(exe) if exe else "未設定 translate.server_exe")
        model = Path(settings.translate.model_path) if settings.translate.model_path else None
        line(bool(model and model.exists()), "翻譯模型", str(model) if model else "未設定 translate.model_path")
    print(f"  設定檔：{paths.config_path()}")
    print(f"  工作資料夾：{paths.jobs_dir()}")
    from omni2.core import hardware
    from omni2.models.recommend import recommend

    rec = recommend(hardware.detect(paths.data_dir()))
    print(f"\n推薦：{rec.tier}")
    for text in rec.describe() + [f"⚠ {w}" for w in rec.warnings]:
        print(f"  {text}")
    changes = rec.changes(settings)
    print("  目前的設定就是推薦的組合" if not changes else f"  執行 omni2 models recommend --apply 可以套用（{len(changes)} 項設定不同）")
    return 0 if ok else 1


def cmd_models(args, settings: Settings) -> int:
    from omni2.config.settings import save_settings
    from omni2.core import hardware
    from omni2.core.storage import human
    from omni2.models import catalog, download
    from omni2.models.recommend import models_in_use, recommend

    if args.action == "list":
        in_use = models_in_use(settings)
        for m in catalog.CATALOG:
            installed = catalog.is_installed(m)
            size = human(catalog.installed_size(m)) if installed else f"約 {human(m.size)}"
            mark = "✓" if installed else "·"
            version = catalog.installed_version(m)
            print(f"{mark} {m.id:<16} {m.label + (f'（{version}）' if version else ''):<28} {size:>10}"
                  f"{'  使用中' if m.id in in_use else ''}")
        return 0

    hw = hardware.detect(paths.data_dir())
    if args.action == "recommend":
        rec = recommend(hw)
        print(rec.tier)
        for text in rec.describe() + [f"⚠ {w}" for w in rec.warnings]:
            print(f"  {text}")
        for label, old, new in rec.changes(settings):
            print(f"  {label}：{old} → {new}")
        if args.apply:
            rec.apply(settings)
            save_settings(settings)
            print("已套用。")
        ids = [i for i in rec.models() if not catalog.is_installed(catalog.get(i))]
        if ids:
            print(f"還沒下載：{' '.join(ids)}（omni2 models download {' '.join(ids)}）")
        return 0

    for model_id in args.ids:  # download
        if model_id not in catalog.BY_ID:
            print(f"沒有這個模型：{model_id}（omni2 models list 可以看所有模型）")
            return 2
    for model_id in args.ids:
        m = catalog.get(model_id)
        print(f"下載 {m.label}…", flush=True)

        def progress(done: int, total: int, text: str) -> None:
            line = f"  {human(done)} / {human(total)}" if total else f"  {text}"
            sys.stdout.write("\r" + line.ljust(40))
            sys.stdout.flush()

        result = download.download(model_id, progress, driver_cuda=hw.driver_cuda)
        print(f"\n  完成：{result.get('path', '')}")
        if model_id == "llama-server":
            settings.translate.server_exe = result["path"]
            save_settings(settings)
    return 0


def cmd_shortcut(args) -> int:
    from omni2.core import shortcut

    folders = {"desktop": shortcut.desktop_dir(), "start": shortcut.start_menu_dir()}
    for where in args.where:
        print(f"已建立：{shortcut.create(folders[where])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mimiyaku", description="Mimiyaku")
    parser.add_argument("-v", "--verbose", action="store_true", help="顯示詳細記錄")
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("file", help="翻譯或轉錄音檔、影片")
    f.add_argument("inputs", nargs="+", help="檔案路徑或影片網址（可多個）")
    f.add_argument("--to", dest="tgt", help="目標語言，例如 zh-TW")
    f.add_argument("--from", dest="src", help="來源語言，例如 ja；不填就自動偵測")
    f.add_argument("--no-translate", action="store_true", help="純紀錄：只辨識不翻譯")
    f.add_argument("--formats", help=f"輸出格式，逗號分隔：{','.join(ALL_FORMATS)}")
    f.add_argument("--bilingual", action="store_true", help="字幕同時顯示譯文和原文")
    f.add_argument("--out", help="輸出資料夾（預設和原檔同資料夾）")
    f.add_argument("--stt", choices=["qwen3-asr", "faster-whisper", "funasr", "openai", "google"],
                   help="STT 引擎（funasr、openai、google 是雲端服務，需要金鑰）")
    f.add_argument("--stt-model", help="STT 模型，例如 large-v3-turbo、Qwen/Qwen3-ASR-0.6B")
    f.add_argument("--vad", choices=list(PRESETS), help="VAD 情境預設")
    f.add_argument("--model", help="翻譯用的 GGUF 檔")
    f.add_argument("--glossary", help="額外的術語表 YAML")
    f.add_argument("--vocals", action="store_true", help="先做人聲分離（有背景音樂時）")
    f.add_argument("--diarize", action="store_true", help="講者分離")
    f.add_argument("--speakers", type=int, help="講者人數（已知時指定，比自動判斷準）")
    f.add_argument("--terms", action="store_true", help="全文術語先行：翻譯前先整理摘要和專有名詞")
    f.add_argument("--no-review", action="store_true", help="不確認，直接採用抽出來的術語")
    f.add_argument("--jobs-dir", help=argparse.SUPPRESS)

    sub.add_parser("doctor", help="檢查環境、推薦模型組合")
    m = sub.add_parser("models", help="模型：列出、推薦、下載")
    m.add_argument("action", choices=["list", "recommend", "download"])
    m.add_argument("ids", nargs="*", help="download 要下載的模型 id（見 list）")
    m.add_argument("--apply", action="store_true", help="recommend：套用推薦設定")
    s = sub.add_parser("shortcut", help="建立啟動捷徑")
    s.add_argument("where", nargs="+", choices=["desktop", "start"], help="desktop＝桌面，start＝開始功能表")
    c = sub.add_parser("config", help="設定檔")
    c.add_argument("action", choices=["path", "show"])
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    settings = load_settings()
    if args.cmd == "file":
        return cmd_file(args, settings)
    if args.cmd == "doctor":
        return cmd_doctor(settings)
    if args.cmd == "models":
        return cmd_models(args, settings)
    if args.cmd == "shortcut":
        return cmd_shortcut(args)
    if args.action == "path":
        print(paths.config_path())
    else:
        print(paths.config_path().read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
