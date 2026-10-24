"""講者分離：每句算一個 ECAPA 聲紋，再用階層式聚類把聲紋相近的句子歸給同一位講者。

不用 pyannote：它的模型需要 Hugging Face 帳號同意條款才能下載。這個做法不需要登入，
適合會議、訪談這類輪流發言的內容；兩人同時說話時分不開。
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import numpy as np

from omni2.audio.decode import SAMPLE_RATE, read_wav
from omni2.core import paths
from omni2.core.jsonio import read_json, write_json_atomic

log = logging.getLogger(__name__)

# 太長的句子只取中間這麼多秒算聲紋，夠穩定又不浪費時間
_MAX_EMBED_S = 10.0


def cluster(embeddings: np.ndarray, threshold: float, num_speakers: int | None = None) -> list[int]:
    """平均連結的階層式聚類（餘弦距離）。回傳的編號依第一次出現的順序排列。"""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]
    x = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    tree = linkage(x, method="average", metric="cosine")
    if num_speakers:
        raw = fcluster(tree, t=num_speakers, criterion="maxclust")
    else:
        raw = fcluster(tree, t=threshold, criterion="distance")
    order: dict[int, int] = {}
    return [order.setdefault(int(label), len(order)) for label in raw]


def assign_short(labels: list[int | None], embeddings: list[np.ndarray | None]) -> list[int]:
    """沒參與聚類的短句：有聲紋就歸給最像的講者，沒有就沿用前一句（第一句則用下一句）。"""
    known = [(lab, emb) for lab, emb in zip(labels, embeddings) if lab is not None and emb is not None]
    centroids: dict[int, np.ndarray] = {}
    for lab in {lab for lab, _ in known}:
        vecs = np.stack([e for l, e in known if l == lab])
        c = vecs.mean(axis=0)
        centroids[lab] = c / np.linalg.norm(c)
    out: list[int | None] = list(labels)
    for i, (lab, emb) in enumerate(zip(labels, embeddings)):
        if lab is None and emb is not None and centroids:
            v = emb / np.linalg.norm(emb)
            out[i] = max(centroids, key=lambda k: float(v @ centroids[k]))
    for i in range(len(out)):
        if out[i] is None and i > 0:
            out[i] = out[i - 1]
    for i in range(len(out) - 1, -1, -1):
        if out[i] is None:
            out[i] = out[i + 1] if i + 1 < len(out) else 0
    return [int(x) for x in out]


def run(payload: dict, emit) -> dict:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    model = payload["model"]
    emit({"type": "status", "message": f"載入聲紋模型 {model}"})
    save_dir = paths.app_dir() / "models" / "speechbrain" / model.replace("/", "--")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = EncoderClassifier.from_hparams(
        source=model, savedir=str(save_dir), run_opts={"device": device},
        # Windows 預設不允許一般使用者建立 symlink
        local_strategy=LocalStrategy.COPY,
    )
    threshold, min_segment = payload["threshold"], payload["min_segment"]

    for item in payload["items"]:
        job = item["job"]
        try:
            emit({"type": "job_start", "job": job})
            audio = read_wav(Path(item["audio"]))
            segments = read_json(Path(item["transcript"]))["segments"]
            embeddings: list[np.ndarray | None] = []
            for seg in segments:
                start, end = seg["start"], seg["end"]
                if end - start > _MAX_EMBED_S:
                    mid = (start + end) / 2
                    start, end = mid - _MAX_EMBED_S / 2, mid + _MAX_EMBED_S / 2
                clip = audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
                if len(clip) < 0.3 * SAMPLE_RATE:
                    embeddings.append(None)
                    continue
                with torch.no_grad():
                    emb = encoder.encode_batch(torch.from_numpy(clip).unsqueeze(0).to(device))
                embeddings.append(emb.squeeze().cpu().numpy())
            long_idx = [i for i, s in enumerate(segments)
                        if embeddings[i] is not None and s["end"] - s["start"] >= min_segment]
            labels: list[int | None] = [None] * len(segments)
            if long_idx:
                found = cluster(np.stack([embeddings[i] for i in long_idx]), threshold, item.get("num_speakers"))
                for i, lab in zip(long_idx, found):
                    labels[i] = lab
            final = assign_short(labels, embeddings) if segments else []
            speakers = [f"SPEAKER_{lab:02d}" for lab in final]
            write_json_atomic(Path(item["out"]), {"fp": item["fp"], "speakers": speakers})
            emit({"type": "job_done", "job": job, "speakers": len(set(speakers))})
        except Exception as e:  # noqa: BLE001
            emit({"type": "job_error", "job": job, "message": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()})
    return {"ok": True}
