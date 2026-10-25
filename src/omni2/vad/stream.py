"""串流版的 Silero VAD：每 512 個取樣（32 毫秒）輸出一次人聲機率。

faster-whisper 附的 Silero v6 ONNX 模型本身支援串流，只要把 LSTM 狀態（h、c）和前一塊最後
64 個取樣接著帶進下一次就好。
"""

from __future__ import annotations

import numpy as np

FRAME = 512
CONTEXT = 64


class StreamingVad:
    def __init__(self):
        from faster_whisper.vad import get_vad_model

        self.session = get_vad_model().session
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.context = np.zeros(CONTEXT, np.float32)
        self.pending = np.zeros(0, np.float32)

    def __call__(self, audio: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """回傳這次湊滿的每一格：（512 個取樣, 人聲機率）。不滿一格的留到下次。"""
        self.pending = np.concatenate([self.pending, audio.astype(np.float32, copy=False)])
        out = []
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            x = np.concatenate([self.context, frame])[None, :]
            prob, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
            self.context = frame[-CONTEXT:]
            out.append((frame, float(np.asarray(prob).reshape(-1)[0])))
        return out
