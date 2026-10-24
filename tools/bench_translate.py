"""比較翻譯模型：同一份評測句子，各模型用各自的 prompt 風格翻譯，輸出分數和逐句對照。

    uv run python tools/bench_translate.py --out samples/bench \\
        qwen3=D:/models/Qwen3-8B-Q4_K_M.gguf \\
        hymt2=C:/.../HY-MT2-7B-Q6_K.gguf \\
        hymt2-generic=C:/.../HY-MT2-7B-Q6_K.gguf:generic

每個參數是「名稱=GGUF路徑[:prompt風格[@溫度]]」，不寫風格就依檔名自動判斷。
指向同一個 GGUF 的設定會共用同一個 llama-server。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

from omni2.config.settings import load_settings
from omni2.translate.client import ChatClient
from omni2.translate.prompts import style_name_for
from omni2.translate.server import LlamaServer
from omni2.translate.translator import BatchTranslator, make_converter

DATA = Path(__file__).parent / "bench_data" / "translate.json"
TARGET = "zh-TW"


def _norm(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def chrf(hyps: list[str], refs: list[str], n: int = 6, beta: float = 2.0) -> float:
    """語料層級的 chrF（字元 n-gram 的 F 分數，0–100）。"""
    match, hyp_total, ref_total = [0] * n, [0] * n, [0] * n
    for h, r in zip(hyps, refs):
        h, r = _norm(h), _norm(r)
        for k in range(1, n + 1):
            hc = Counter(h[i : i + k] for i in range(len(h) - k + 1))
            rc = Counter(r[i : i + k] for i in range(len(r) - k + 1))
            match[k - 1] += sum((hc & rc).values())
            hyp_total[k - 1] += sum(hc.values())
            ref_total[k - 1] += sum(rc.values())
    ks = [k for k in range(n) if hyp_total[k] and ref_total[k]]
    if not ks:
        return 0.0
    p = sum(match[k] / hyp_total[k] for k in ks) / len(ks)
    r = sum(match[k] / ref_total[k] for k in ks) / len(ks)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


def vram_used_mb() -> int | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    out = subprocess.run([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else None


def run_config(client: ChatClient, style: str, data: dict, temperature: float | None = None) -> dict:
    convert = make_converter("s2twp", TARGET)
    result = {"style": style if temperature is None else f"{style}@{temperature}", "langs": {}}
    for src_lang, scenes in data.items():
        outs, refs, srcs, issues, fixed, seconds = [], [], [], 0, 0, 0.0
        for scene in scenes:
            lines = [src for src, _ in scene["lines"]]
            t = BatchTranslator(client, src_lang=src_lang, tgt_lang=TARGET, style=style, opencc="")
            if temperature is not None:
                t.temperature = temperature
            start = time.perf_counter()
            raw = t.translate_all(lines)
            seconds += time.perf_counter() - start
            issues += len(t.issues)
            for line in raw:
                converted = convert(line)
                fixed += sum(a != b for a, b in zip(line, converted))
                outs.append(converted)
            srcs += lines
            refs += [ref for _, ref in scene["lines"]]
        result["langs"][src_lang] = {
            "chrf": round(chrf(outs, refs), 1),
            "issues": issues,
            "opencc_fixed_chars": fixed,
            "seconds": round(seconds, 1),
            "lines": [{"src": s, "ref": r, "out": o} for s, r, o in zip(srcs, refs, outs)],
        }
    return result


def report(results: dict[str, dict], out: Path) -> str:
    langs = list(next(iter(results.values()))["langs"])
    head = "| 設定 | 風格 | " + " | ".join(f"chrF {l}" for l in langs) + " | 平均 | 未過檢查 | OpenCC 修正字數 | 秒／100 句 | VRAM |"
    rows = [head, "|" + "---|" * (head.count("|") - 1)]
    for name, r in results.items():
        ls = r["langs"].values()
        n_lines = sum(len(x["lines"]) for x in ls)
        avg = sum(x["chrf"] for x in ls) / len(langs)
        speed = 100 * sum(x["seconds"] for x in ls) / n_lines
        rows.append(
            f"| {name} | {r['style']} | " + " | ".join(f"{r['langs'][l]['chrf']:.1f}" for l in langs)
            + f" | **{avg:.1f}** | {sum(x['issues'] for x in ls)} | {sum(x['opencc_fixed_chars'] for x in ls)}"
            + f" | {speed:.1f} | {r.get('vram_mb', '?')} MB |"
        )
    md = ["# 翻譯模型比較", "", *rows, ""]
    names = list(results)
    for lang in langs:
        md += [f"## {lang} → {TARGET}", ""]
        first = results[names[0]]["langs"][lang]["lines"]
        for i, line in enumerate(first):
            md.append(f"**{i + 1}. {line['src']}**  ")
            md.append(f"參考：{line['ref']}  ")
            for name in names:
                md.append(f"`{name}`：{results[name]['langs'][lang]['lines'][i]['out']}  ")
            md.append("")
    text = "\n".join(md)
    (out / "report.md").write_text(text, encoding="utf-8")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="名稱=GGUF路徑[:風格]")
    parser.add_argument("--out", default="samples/bench")
    parser.add_argument("--langs", help="只測某些來源語言，例如 ja,ko")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data = json.loads(DATA.read_text(encoding="utf-8"))
    if args.langs:
        data = {k: v for k, v in data.items() if k in args.langs.split(",")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = load_settings().translate

    by_model: dict[str, list[tuple[str, str, float | None]]] = {}
    for spec in args.configs:
        name, rest = spec.split("=", 1)
        path, style = rest, "auto"
        if rest.count(":") > 1 or (rest.count(":") == 1 and len(rest.split(":")[0]) > 1):
            path, style = rest.rsplit(":", 1)
        style, _, temp = style.partition("@")
        by_model.setdefault(path, []).append((name, style_name_for(path, style), float(temp) if temp else None))

    results: dict[str, dict] = {}
    for path, configs in by_model.items():
        print(f"» 載入 {Path(path).name}", flush=True)
        before = vram_used_mb()
        server = LlamaServer(settings.server_exe, path, ctx_size=8192, gpu_layers=99)
        start = time.perf_counter()
        client = ChatClient(server.start())
        load_s = time.perf_counter() - start
        after = vram_used_mb()
        try:
            for name, style, temp in configs:
                print(f"  {name}（{style}）", flush=True)
                r = run_config(client, style, data, temp)
                r["load_seconds"] = round(load_s, 1)
                r["vram_mb"] = (after - before) if before is not None and after is not None else None
                results[name] = r
                (out / f"{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        finally:
            client.close()
            server.stop()
    print(report(results, out))
    print(f"\n逐句對照：{out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
