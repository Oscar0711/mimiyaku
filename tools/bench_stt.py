"""比較 STT 引擎：對 samples/*.wav 算字元錯誤率（CER），參考答案是 tools/make_samples.py 產生的 *.ref.json。

    uv run python tools/bench_stt.py faster-whisper qwen3-asr [--samples samples] [--only stream_ja,talk_en]

雲端引擎（funasr、openai、google）會用設定檔裡的雲端設定和認證管理員裡的金鑰，並且會計費，
建議搭配 --only 只跑少數樣本。

CER 會先做 NFKC、去掉標點和空白；漢字和阿拉伯數字的寫法差異（三回／3回）也會算成錯誤，
所以只適合拿來比較引擎之間的相對好壞。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

from omni2.audio.decode import read_wav
from omni2.config.settings import load_settings
from omni2.stt.base import TranscribeOptions
from omni2.stt.registry import create_engine
from omni2.vad.presets import get_preset


def norm(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text) if c.isalnum()).casefold()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engines", nargs="+")
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--only", help="只跑這些樣本（逗號分隔，不含副檔名）")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    wavs = sorted(p for p in Path(args.samples).glob("*.wav") if p.with_suffix(".ref.json").exists())
    if args.only:
        wanted = set(args.only.split(","))
        wavs = [p for p in wavs if p.stem in wanted]
    base = load_settings().stt
    audios = [read_wav(p) for p in wavs]
    refs = [json.loads(p.with_suffix(".ref.json").read_text(encoding="utf-8")) for p in wavs]
    rows = []
    for name in args.engines:
        engine = create_engine(base.model_copy(update={"engine": name}))
        start = time.perf_counter()
        engine.load()
        load_s = time.perf_counter() - start
        start = time.perf_counter()
        transcripts = engine.transcribe_many(audios, TranscribeOptions(vad=get_preset("file")))
        run_s = time.perf_counter() - start
        errors = total = 0
        for wav, ref, tr in zip(wavs, refs, transcripts):
            r = norm("".join(x["text"] for x in ref))
            h = norm("".join(s.text for s in tr.segments))
            e = edit_distance(r, h)
            errors += e
            total += len(r)
            print(f"[{name}] {wav.stem}: CER {e / len(r):.3f}，{len(tr.segments)} 句")
            for s in tr.segments:
                print(f"    {s.start:6.2f}-{s.end:6.2f} {s.text}")
        rows.append((name, errors / total, load_s, run_s))
        engine.close()
        del engine
    print("\n| 引擎 | CER | 載入秒數 | 辨識秒數 |\n|---|---|---|---|")
    for name, cer, load_s, run_s in rows:
        print(f"| {name} | {cer:.3f} | {load_s:.1f} | {run_s:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
