"""人聲分離（audio-separator），在獨立子行程裡執行。

分離模型用 44.1k 立體聲訓練，所以從原始檔重新解碼，不拿已經降成 16k 單聲道的 audio.wav。
分離出的人聲再轉成 16k 單聲道 vocals.wav 給 STT。
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import traceback
from pathlib import Path

from omni2.audio.decode import decode_to_wav
from omni2.core import paths

log = logging.getLogger(__name__)


def run(payload: dict, emit) -> dict:
    from audio_separator.separator import Separator

    model = payload["model"]
    emit({"type": "status", "message": f"載入人聲分離模型 {model}"})
    model_dir = paths.models_dir() / "separator"
    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="omni2-sep-") as tmp:
        sep = Separator(
            log_level=logging.WARNING,
            model_file_dir=str(model_dir),
            output_dir=tmp,
            output_single_stem="Vocals",
        )
        sep.load_model(model_filename=model)
        for item in payload["items"]:
            job = item["job"]
            try:
                emit({"type": "job_start", "job": job})
                mix = Path(tmp) / f"{job}.mix.wav"
                decode_to_wav(Path(item["input"]), mix, sr=44100, channels=2)
                outputs = sep.separate(str(mix), custom_output_names={"Vocals": f"{job}.vocals"})
                vocals = Path(tmp) / Path(outputs[0]).name
                decode_to_wav(vocals, Path(item["out"]))
                mix.unlink(missing_ok=True)
                vocals.unlink(missing_ok=True)
                emit({"type": "job_done", "job": job})
            except Exception as e:  # noqa: BLE001 — 單一檔案失敗不影響其他檔案
                emit({"type": "job_error", "job": job, "message": f"{type(e).__name__}: {e}",
                      "traceback": traceback.format_exc()})
        shutil.rmtree(tmp, ignore_errors=True)
    return {"ok": True}
