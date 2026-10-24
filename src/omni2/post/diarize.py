"""講者分離：找出「誰在什麼時間說話」，再把每個詞分給講者，在換人的地方把句子切開。

兩個引擎：
* ecapa（預設，不需帳號）：在語音上滑動 1.5 秒的窗口，每個窗口算一個 SpeechBrain ECAPA
  聲紋，用平均連結的階層式聚類分群。以前是一整句算一個聲紋，但短回應常和別人的話併在
  同一句裡，聲紋被稀釋，實測真實錄音大多被分成同一人。
* pyannote：pyannote/speaker-diarization-community-1，比較準，但模型需要 Hugging Face 帳號
  同意條款，並在這台電腦登入（hf auth login）。

兩者都產生講者時間軸（Turn），後面共用逐詞判斷、平滑、切句的邏輯。
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from omni2.audio.decode import SAMPLE_RATE, read_wav
from omni2.core import paths
from omni2.core.jsonio import read_json, write_json_atomic
from omni2.core.types import Segment, Transcript, Word

log = logging.getLogger(__name__)

# 第一階段（分群）：長窗口的聲紋比較穩定，用來找出有哪幾位講者
WINDOW_S = 1.5
HOP_S = 0.75
MIN_WINDOW_S = 0.5  # 比這短的語音不算聲紋
# 第二階段（細分）：已知每位講者的聲紋後，用短窗口逐段比對最像誰，換人的時間點可以精確到約 0.25 秒。
# 只用長窗口時，跨在換人處的窗口同時有兩個人的聲音，快速接話時切點會偏掉半秒以上。
FINE_WINDOW_S = 0.75
FINE_HOP_S = 0.25
FINE_MIN_S = 0.3
MIN_RUN_S = 0.3  # 比這短的換人視為誤判，併回前後的講者（以前 0.6 秒，會吞掉「了解。」這種短回應）
SNAP_WORDS = 2  # 換人的切點最多往前後挪幾個詞，去對齊停頓或句尾標點
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass(slots=True)
class Turn:
    start: float
    end: float
    speaker: int


# ---- 聲紋與聚類（ecapa） ----

UNIT_PAUSE_S = 0.3  # 詞與詞之間停頓這麼久，視為可能換人的地方
_SENTENCE_END = ("。", "！", "？", "!", "?", "…", ".")
# 切句時逗號也算：只是讓窗口不跨過，不會強迫換人。實測快速接話 88% → 93%，其他樣本不變。
_UNIT_BREAK = _SENTENCE_END + ("、", "，", ",")


def speech_units(segments: list[Segment]) -> list[tuple[float, float]]:
    """把每一段再依標點（句尾和逗號）和停頓切成「一句一句」。

    快速接話時，VAD 會把好幾個人的話併成同一段，每人一次只講約 1 秒；窗口如果跨過句子，
    算出來的是兩個人混在一起的聲紋，分群就會亂（實測三人快速接話只有 67%、多分出一人）。
    換人幾乎都發生在句尾或停頓，所以窗口不跨過這些地方。
    """
    units = []
    for seg in segments:
        if not seg.words:
            units.append((seg.start, seg.end))
            continue
        start = seg.words[0].start
        for i, w in enumerate(seg.words):
            last = i + 1 == len(seg.words)
            ends = w.text.strip().endswith(_UNIT_BREAK)
            pause = not last and seg.words[i + 1].start - w.end >= UNIT_PAUSE_S
            if last or ends or pause:
                units.append((start, w.end))
                if not last:
                    start = seg.words[i + 1].start
    return units


def speech_windows(spans: list[tuple[float, float]], window: float = WINDOW_S, hop: float = HOP_S,
                   min_len: float = MIN_WINDOW_S) -> list[tuple[float, float]]:
    """在每個範圍內切出重疊的窗口，不跨過範圍邊界；短的範圍整個一個窗口。"""
    windows = []
    for start, end in spans:
        span = end - start
        if span < min_len:
            continue
        if span <= window:
            windows.append((start, end))
            continue
        t = start
        while t + window < end:
            windows.append((t, t + window))
            t += hop
        windows.append((max(start, end - window), end))
    return windows


def speaker_centroids(embeddings: np.ndarray, labels: list[int]) -> np.ndarray:
    """每位講者的聲紋中心。優先只用前後鄰居也是同一人的「乾淨」窗口：跨在換人處的窗口
    同時有兩個人的聲音，會把中心拉偏。"""
    x = _normalize(embeddings)
    lab = np.asarray(labels)
    pure = np.array([(i == 0 or lab[i - 1] == lab[i]) and (i + 1 == len(lab) or lab[i + 1] == lab[i])
                     for i in range(len(lab))])
    cents = []
    for k in range(lab.max() + 1):
        mask = (lab == k) & pure
        if mask.sum() < 2:
            mask = lab == k
        cents.append(x[mask].mean(axis=0))
    return _normalize(np.stack(cents))


def classify_windows(embeddings: np.ndarray, centroids: np.ndarray,
                     windows: list[tuple[float, float]]) -> list[int]:
    """每個短窗口歸給最像的講者。比對分數先和時間上相鄰（重疊）的窗口平均，降低單一窗口的誤判。"""
    sims = _normalize(embeddings) @ centroids.T
    smoothed = sims.copy()
    for i in range(len(windows)):
        for j in (i - 1, i + 1):
            if 0 <= j < len(windows) and windows[j][0] < windows[i][1] and windows[i][0] < windows[j][1]:
                smoothed[i] += 0.5 * sims[j]
    return [int(v) for v in np.argmax(smoothed, axis=1)]


# 預設的聚類門檻（平均連結的餘弦距離）。合成語音樣本上，多人之間的合併距離多在 0.76 以上，
# 單人的大群在 0.7 以內（離群的小群由 absorb_small 併回）；見 tools/bench_diarize.py。
DEFAULT_THRESHOLD = 0.7
# 窗口太多時（長音檔）先抽樣聚類，其餘的窗口再歸給最近的講者
MAX_CLUSTER_POINTS = 2500


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cluster(embeddings: np.ndarray, threshold: float = DEFAULT_THRESHOLD, num_speakers: int | None = None,
            min_share: float = 0.05) -> list[int]:
    """平均連結的階層式聚類（餘弦距離），回傳的編號依第一次出現的順序排列。

    * 沒給人數：在 threshold 切開，再把離群的小群併回大群。
    * 給了人數：從聚類樹最上層往下切，每一層都先併掉小群，直到剛好是這個人數。
      直接切成 N 群的話，一兩個離群窗口（例如大喊、笑聲）就會佔掉一個名額。
    曾試過用輪廓係數自動判斷人數，但單人錄音裡大喊的幾個窗口會自成一群、係數反而很高，不可靠。
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]
    x = _normalize(embeddings)
    if n > MAX_CLUSTER_POINTS:
        idx = np.linspace(0, n - 1, MAX_CLUSTER_POINTS).astype(int)
        sub = np.asarray(cluster(embeddings[idx], threshold, num_speakers, min_share))
        centroids = _normalize(np.stack([x[idx][sub == k].mean(axis=0) for k in range(sub.max() + 1)]))
        return _relabel([int(i) for i in np.argmax(x @ centroids.T, axis=1)])

    tree = linkage(x, method="average", metric="cosine")

    def cut(t: float) -> list[int]:
        raw = _relabel([int(r) for r in fcluster(tree, t=t, criterion="distance")])
        return absorb_small(raw, embeddings, min_share)

    if not num_speakers:
        return cut(threshold)
    # 以預設門檻為起點：人太多就合併最像的群，人太少就往下切到第一個夠的地方。
    # 從聚類樹頂端或底端去湊人數都試過：頂端湊到的第 N 群常是一小撮特別的窗口，
    # 底端則會把很多碎群硬併，兩者在合成樣本上都掉到 60～80%。
    labels = cut(threshold)
    if len(set(labels)) < num_speakers:
        for height in sorted((h for h in set(tree[:, 2]) if h < threshold), reverse=True):
            labels = cut(height - 1e-9)
            if len(set(labels)) >= num_speakers:
                break
    return _merge_to(labels, x, num_speakers)


def _merge_to(labels: list[int], x: np.ndarray, k: int) -> list[int]:
    """反覆合併質心最像的兩群，直到剩 k 群。x 需已正規化。"""
    lab = np.asarray(labels)
    while len(np.unique(lab)) > k:
        ks = np.unique(lab)
        cents = _normalize(np.stack([x[lab == c].mean(axis=0) for c in ks]))
        sim = cents @ cents.T
        np.fill_diagonal(sim, -np.inf)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        lab[lab == ks[j]] = ks[i]
    return _relabel([int(v) for v in lab])


def _relabel(labels: list[int]) -> list[int]:
    order: dict[int, int] = {}
    return [order.setdefault(label, len(order)) for label in labels]


def absorb_small(labels: list[int], embeddings: np.ndarray, min_share: float) -> list[int]:
    """窗口數佔比低於 min_share 的群多半是雜訊或換氣，併到聲紋最像的大群。"""
    counts = np.bincount(labels)
    big = [k for k in range(len(counts)) if counts[k] >= max(2, min_share * len(labels))]
    if not big:  # 全部都是小群（幾乎每個窗口自成一群）：不處理，交給呼叫端
        return labels
    if len(big) == len(counts):
        return labels
    x = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    centroids = {k: x[[i for i, lab in enumerate(labels) if lab == k]].mean(axis=0) for k in big}
    out = [lab if lab in centroids else max(centroids, key=lambda k: float(x[i] @ centroids[k]))
           for i, lab in enumerate(labels)]
    return _relabel(out)


def windows_to_turns(windows: list[tuple[float, float]], labels: list[int]) -> list[Turn]:
    """窗口重疊，所以每個窗口只負責它中心附近的時間；相鄰同講者合併。"""
    turns: list[Turn] = []
    for k, ((start, end), lab) in enumerate(zip(windows, labels)):
        lo = start if k == 0 or windows[k - 1][1] <= start else (start + windows[k - 1][1]) / 2
        hi = end if k + 1 == len(windows) or windows[k + 1][0] >= end else (end + windows[k + 1][0]) / 2
        if turns and turns[-1].speaker == lab and lo - turns[-1].end < 0.05:
            turns[-1].end = hi
        else:
            turns.append(Turn(lo, hi, lab))
    return turns


# ---- 逐詞判斷、切句（兩個引擎共用） ----

def _speaker_at(turns: list[Turn], start: float, end: float) -> int | None:
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = min(end, t.end) - max(start, t.start)
        if overlap > best_overlap:
            best, best_overlap = t.speaker, overlap
    if best is not None:
        return best
    if not turns:
        return None
    mid = (start + end) / 2
    return min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).speaker


def _smooth(labels: list[int], words: list[Word]) -> list[int]:
    """把持續不到 MIN_RUN_S 的講者片段併回前一段（開頭則併到下一段）。"""
    runs: list[list[int]] = []  # [講者, 起始詞, 結束詞（不含）]
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i + 1
        else:
            runs.append([lab, i, i + 1])
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k, (lab, a, b) in enumerate(runs):
            if words[b - 1].end - words[a].start < MIN_RUN_S:
                runs[k][0] = runs[k - 1][0] if k > 0 else runs[k + 1][0]
                merged: list[list[int]] = []
                for r in runs:
                    if merged and merged[-1][0] == r[0]:
                        merged[-1][2] = r[2]
                    else:
                        merged.append(list(r))
                runs = merged
                changed = True
                break
    out = []
    for lab, a, b in runs:
        out += [lab] * (b - a)
    return out


def _boundary_score(words: list[Word], i: int) -> float:
    """在第 i 個詞前面換人有多自然：停頓越長越好，前一個詞是句尾標點更好。"""
    pause = max(0.0, words[i].start - words[i - 1].end)
    return pause + (0.3 if words[i - 1].text.strip().endswith(_SENTENCE_END) else 0.0)


def _snap(labels: list[int], words: list[Word]) -> list[int]:
    """把換人的切點挪到附近（前後 SNAP_WORDS 個詞內）停頓最明顯或句尾的位置。
    聲紋窗口有時間解析度的限制，快速接話時切點常差一兩個詞（例如「買おう｜か」）。"""
    out = list(labels)
    i = 1
    while i < len(out):
        if out[i] == out[i - 1]:
            i += 1
            continue
        left, right = out[i - 1], out[i]
        # 只在兩邊各自連續的範圍內挪，不跨過另一個換人點
        lo = i - 1
        while lo > 1 and i - lo < SNAP_WORDS and out[lo - 1] == left:
            lo -= 1
        hi = i + 1
        while hi < len(out) and hi - i < SNAP_WORDS and out[hi] == right:
            hi += 1
        best = max(range(max(lo, 1), hi), key=lambda j: (_boundary_score(words, j), -abs(j - i)))
        if _boundary_score(words, best) > _boundary_score(words, i) + 0.05:
            for j in range(min(best, i), max(best, i)):
                out[j] = left if best > i else right
        i = max(best, i) + 1
    return out


def assign_speakers(segments: list[Segment], turns: list[Turn]) -> list[Segment]:
    """每個詞分給講者，在換人的地方把句子切開。回傳的新片段帶 speaker（SPEAKER_00…）。"""
    out: list[Segment] = []
    for seg in segments:
        if not seg.words:
            lab = _speaker_at(turns, seg.start, seg.end)
            out.append(_copy(seg, seg.words, seg.start, seg.end, seg.text, lab))
            continue
        labels = [_speaker_at(turns, w.start, w.end) for w in seg.words]
        known = [lab for lab in labels if lab is not None]
        fallback = known[0] if known else 0
        labels = _smooth([fallback if lab is None else lab for lab in labels], seg.words)
        labels = _snap(labels, seg.words)
        start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start]:
                words = seg.words[start:i]
                lo = seg.start if start == 0 else words[0].start
                hi = seg.end if i == len(labels) else words[-1].end
                text = "".join(w.text for w in words).strip()
                out.append(_copy(seg, words, lo, hi, text, labels[start]))
                start = i
    names = {lab: f"SPEAKER_{k:02d}" for k, lab in enumerate(dict.fromkeys(
        s.speaker for s in out if s.speaker is not None))}
    for s in out:
        s.speaker = names.get(s.speaker, "SPEAKER_00")
    return out


def _copy(seg: Segment, words: list[Word], start: float, end: float, text: str, speaker) -> Segment:
    return Segment(start=start, end=end, text=text, words=list(words), speaker=speaker,
                   no_speech_prob=seg.no_speech_prob, avg_logprob=seg.avg_logprob)


# ---- 引擎 ----

class EcapaDiarizer:
    def __init__(self, model: str, threshold: float = DEFAULT_THRESHOLD, min_share: float = 0.05,
                 refine: bool = True):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        save_dir = paths.models_dir() / "speechbrain" / model.replace("/", "--")
        self.encoder = EncoderClassifier.from_hparams(
            source=model, savedir=str(save_dir), run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,  # Windows 預設不允許一般使用者建立 symlink
        )
        self.threshold = threshold
        self.min_share = min_share
        self.refine = refine

    def embed(self, audio: np.ndarray, windows: list[tuple[float, float]], batch: int = 64) -> np.ndarray:
        import torch

        out = []
        for pos in range(0, len(windows), batch):
            chunk = [audio[int(s * SAMPLE_RATE): int(e * SAMPLE_RATE)] for s, e in windows[pos: pos + batch]]
            longest = max(len(c) for c in chunk)
            wavs = torch.zeros(len(chunk), longest)
            for i, c in enumerate(chunk):
                wavs[i, : len(c)] = torch.from_numpy(c)
            lens = torch.tensor([len(c) / longest for c in chunk])
            with torch.no_grad():
                emb = self.encoder.encode_batch(wavs.to(self.device), lens.to(self.device))
            out.append(emb.squeeze(1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 192))

    def turns(self, audio: np.ndarray, segments: list[Segment], num_speakers: int | None) -> list[Turn]:
        units = speech_units(segments)
        windows = speech_windows(units)
        if not windows:
            return []
        emb = self.embed(audio, windows)
        labels = cluster(emb, self.threshold, num_speakers, self.min_share)
        if not self.refine or max(labels) == 0:
            return windows_to_turns(windows, labels)
        # 第二階段：用短窗口逐段比對每位講者的聲紋中心
        fine = speech_windows(units, FINE_WINDOW_S, FINE_HOP_S, FINE_MIN_S)
        fine_emb = self.embed(audio, fine)
        fine_labels = classify_windows(fine_emb, speaker_centroids(emb, labels), fine)
        if not num_speakers:
            # 第一階段裡雜訊造成的小群，在短窗口比對時會被放大，這裡再併一次
            fine_labels = absorb_small(fine_labels, fine_emb, self.min_share)
        return windows_to_turns(fine, fine_labels)


class PyannoteDiarizer:
    def __init__(self):
        import torch
        from pyannote.audio import Pipeline

        try:
            # token 由 huggingface_hub 從本機登入狀態（hf auth login）讀取
            self.pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "無法下載 pyannote 模型：請先在 Hugging Face 同意 "
                f"{PYANNOTE_MODEL} 的使用條款，再在終端機執行 `uv run hf auth login` 登入。"
                f"（{type(e).__name__}: {e}）"
            ) from e
        if self.pipeline is None:
            raise RuntimeError(f"無法載入 {PYANNOTE_MODEL}：請確認已同意使用條款並登入 Hugging Face")
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

    def turns(self, audio: np.ndarray, segments: list[Segment], num_speakers: int | None) -> list[Turn]:
        import torch

        # 直接給波形，不經過 pyannote 的檔案解碼
        waveform = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE}
        kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        output = self.pipeline(waveform, **kwargs)
        annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(
            output, "speaker_diarization", output)
        names: dict[str, int] = {}
        turns = [Turn(seg.start, seg.end, names.setdefault(label, len(names)))
                 for seg, _, label in annotation.itertracks(yield_label=True)]
        return sorted(turns, key=lambda t: t.start)


def make_diarizer(engine: str, model: str, threshold: float):
    if engine == "pyannote":
        return PyannoteDiarizer()
    return EcapaDiarizer(model, threshold)


def run(payload: dict, emit) -> dict:
    engine = payload["engine"]
    emit({"type": "status", "message": f"載入講者分離模型（{engine}）"})
    diarizer = make_diarizer(engine, payload["model"], payload["threshold"])
    for item in payload["items"]:
        job = item["job"]
        try:
            emit({"type": "job_start", "job": job})
            audio = read_wav(Path(item["audio"]))
            transcript = Transcript.from_dict(read_json(Path(item["transcript"])))
            turns = diarizer.turns(audio, transcript.segments, item.get("num_speakers"))
            transcript.segments = assign_speakers(transcript.segments, turns)
            data = transcript.to_dict()
            data["fp"] = item["fp"]
            write_json_atomic(Path(item["out"]), data)
            speakers = len({s.speaker for s in transcript.segments})
            emit({"type": "job_done", "job": job, "speakers": speakers, "segments": len(transcript.segments)})
        except Exception as e:  # noqa: BLE001
            emit({"type": "job_error", "job": job, "message": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()})
    return {"ok": True}
