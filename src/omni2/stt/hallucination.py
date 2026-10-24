"""過濾 Whisper 從靜音裡「聽出來」的句子。

Whisper 用字幕影片訓練，遇到靜音、噪音或音樂時不會回傳空白，而是回傳訓練資料裡最常
伴隨安靜片段出現的句子：頻道結尾語、字幕組署名、「謝謝觀看」。這些句子的信心分數看起來
很正常，單看文字分辨不出來，所以要明確過濾。

三個訊號，由便宜到昂貴：片語黑名單、no_speech_prob、壓縮比（重複的文字壓縮率高）。
門檻和下面的實測數字沿用上一版。
"""

from __future__ import annotations

import re
import zlib

from omni2.core.types import Segment

_PHRASES = {
    # 英文
    "thank you", "thank you for watching", "thanks for watching", "thank you very much",
    "please subscribe", "subscribe to my channel", "you", "bye", "okay",
    # 字幕署名，最常見的一類
    "字幕由amara org社群提供", "字幕由amaraorg社群提供", "字幕志愿者李宗盛",
    "中文字幕由中文字幕组提供", "本字幕由字幕组提供",
    # 中文頻道結尾語
    "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目", "请不吝点赞订阅转发打赏支持明镜与点点栏目",
    "謝謝觀看", "谢谢观看", "謝謝大家", "下次再見", "明鏡與點點欄目",
    # 日文、韓文
    "ご視聴ありがとうございました", "ご視聴ありがとうございます", "おやすみなさい",
    "시청해주셔서 감사합니다", "구독과 좋아요 부탁드립니다",
}

_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def _normalise(text: str) -> str:
    return _PUNCT_RE.sub("", (text or "").casefold())


_NORMALISED = {_normalise(p) for p in _PHRASES}

# 重複本身不足以定罪時，還要看起來夠安靜。真實語音的 no_speech_prob 約 0.0。
_REPEAT_CORROBORATION_PROB = 0.3


def compression_ratio(text: str) -> float:
    payload = (text or "").encode("utf-8")
    if not payload:
        return 0.0
    return len(payload) / len(zlib.compress(payload))


def hallucination_reason(
    text: str,
    *,
    no_speech_prob: float | None = None,
    ratio: float | None = None,
    no_speech_threshold: float = 0.6,
    max_compression_ratio: float = 2.6,
    extreme_compression_ratio: float = 6.0,
) -> str | None:
    """回傳判定為幻覺的理由；None 表示保留。

    誤判會刪掉真正的語音，所以沒有任何一個訊號能單獨定罪：
    * no_speech_prob 對小聲但真實的語音也會偏高，只有在文字也極短時才丟。
    * 重複也有歧義。實測：一般語音 1.0–1.2、副歌 2.2、同一窗口內真的重複講 3.3–3.8、
      解碼器卡住 8.0。所以偏高的壓縮比要有 no_speech_prob 佐證，只有失控的值能單獨成立。
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    normalised = _normalise(stripped)
    if normalised in _NORMALISED:
        return f"常見幻覺句：{stripped!r}"
    if no_speech_prob is not None and no_speech_prob >= no_speech_threshold and len(normalised) <= 16:
        return f"無語音機率過高（{no_speech_prob:.2f}）且內容極短"
    measured = ratio if ratio is not None else compression_ratio(stripped)
    if len(normalised) >= 24:
        runaway = measured >= extreme_compression_ratio
        repetitive_and_silent = (
            measured >= max_compression_ratio
            and no_speech_prob is not None
            and no_speech_prob >= _REPEAT_CORROBORATION_PROB
        )
        if runaway or repetitive_and_silent:
            return f"內容高度重複（壓縮比 {measured:.2f}）"
    return None


def filter_segments(segments: list[Segment]) -> tuple[list[Segment], list[tuple[Segment, str]]]:
    """回傳（保留的片段, [(被丟掉的片段, 理由)]）。"""
    kept: list[Segment] = []
    dropped: list[tuple[Segment, str]] = []
    for seg in segments:
        if not seg.text.strip():
            dropped.append((seg, "空白"))
            continue
        reason = hallucination_reason(
            seg.text, no_speech_prob=seg.no_speech_prob, ratio=seg.compression_ratio
        )
        if reason:
            dropped.append((seg, reason))
        else:
            kept.append(seg)
    return kept, dropped
