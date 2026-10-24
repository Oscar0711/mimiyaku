"""比較講者分離設定：對有多位講者的樣本，算「時間上標對講者的比例」。

    uv run python tools/bench_diarize.py [--samples samples] [--thresholds 0.3,0.4,0.5,0.6]

參考答案是 tools/make_samples.py 產生的 *.ref.json（每句的講者）。講者名稱不必相同，
會用最佳對應（匈牙利演算法）計分。辨識結果沿用檔案模式的斷點，所以要先跑過 STT。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from omni2.audio.decode import read_wav
from omni2.config.settings import load_settings
from omni2.core.types import Segment
from omni2.jobs.job import JobSpec
from omni2.jobs.runner import FileRunner
from omni2.post.diarize import EcapaDiarizer, Turn, assign_speakers, cluster


def score(ref: list[dict], segments: list[Segment]) -> float:
    refs = sorted({r["speaker"] for r in ref})
    hyps = sorted({s.speaker for s in segments})
    m = np.zeros((len(refs), len(hyps)))
    for r in ref:
        for s in segments:
            overlap = min(r["end"], s.end) - max(r["start"], s.start)
            if overlap > 0:
                m[refs.index(r["speaker"]), hyps.index(s.speaker)] += overlap
    rows, cols = linear_sum_assignment(-m)
    # 分母是「參考和結果重疊的時間」：參考時間含合成語音前後的靜音，不能拿來當分母
    return float(m[rows, cols].sum() / m.sum()) if m.sum() else 0.0


def sentence_level_turns(diarizer: EcapaDiarizer, audio, segments, threshold) -> list[Turn]:
    """舊做法：一整句算一個聲紋（當基準線）。"""
    usable = [s for s in segments if s.end - s.start >= 0.5]
    emb = diarizer.embed(audio, [(s.start, s.end) for s in usable])
    labels = cluster(emb, threshold)
    return [Turn(s.start, s.end, lab) for s, lab in zip(usable, labels)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--thresholds", default="0.6,0.7,0.75")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = load_settings()
    cases = []
    for wav in sorted(Path(args.samples).glob("*.wav")):
        ref_path = wav.with_suffix(".ref.json")
        if not ref_path.exists():
            continue
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        cases.append((wav, ref))

    with tempfile.TemporaryDirectory() as out:
        specs = [JobSpec(input=str(w), translate=False, formats=["txt"], out_dir=out) for w, _ in cases]
        runner = FileRunner(specs, settings)
        runner.run()  # 只會跑還沒辨識過的
        transcripts = [job.load_transcript(runner._fp_stt(job)) for job in runner.jobs]

    diarizer = EcapaDiarizer(settings.diarization.model)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    header = (["樣本", "人數", "舊做法(0.65)"] + [f"只用長窗口 {t}" for t in thresholds]
              + [f"兩階段 {t}" for t in thresholds] + ["兩階段＋已知人數"])
    print("| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for (wav, ref), tr in zip(cases, transcripts):
        audio = read_wav(wav)
        n = len({r["speaker"] for r in ref})
        row = [wav.stem, str(n)]
        old = assign_speakers(tr.segments, sentence_level_turns(diarizer, audio, tr.segments, 0.65))
        row.append(f"{score(ref, old):.0%}（{len({s.speaker for s in old})}人）")
        for refine in (False, True):
            diarizer.refine = refine
            for t in thresholds:
                diarizer.threshold = t
                segs = assign_speakers(tr.segments, diarizer.turns(audio, tr.segments, None))
                row.append(f"{score(ref, segs):.0%}（{len({s.speaker for s in segs})}人）")
        segs = assign_speakers(tr.segments, diarizer.turns(audio, tr.segments, n))
        row.append(f"{score(ref, segs):.0%}（{len({s.speaker for s in segs})}人）")
        print("| " + " | ".join(row) + " |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
