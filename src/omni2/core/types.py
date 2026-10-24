"""整個程式共用的資料型別。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Word:
    start: float
    end: float
    text: str  # 保留引擎給的原始字串（英文會帶前導空白），串起來就是整句
    prob: float | None = None


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    avg_logprob: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d: dict = {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}
        if self.words:
            d["words"] = [
                [round(w.start, 3), round(w.end, 3), w.text, None if w.prob is None else round(w.prob, 3)]
                for w in self.words
            ]
        for key in ("speaker", "no_speech_prob", "compression_ratio", "avg_logprob"):
            value = getattr(self, key)
            if value is not None:
                d[key] = round(value, 4) if isinstance(value, float) else value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        return cls(
            start=d["start"],
            end=d["end"],
            text=d["text"],
            words=[Word(*w) for w in d.get("words", [])],
            speaker=d.get("speaker"),
            no_speech_prob=d.get("no_speech_prob"),
            compression_ratio=d.get("compression_ratio"),
            avg_logprob=d.get("avg_logprob"),
        )


@dataclass(slots=True)
class Transcript:
    segments: list[Segment]
    language: str | None
    duration: float

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "duration": round(self.duration, 3),
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Transcript:
        return cls(
            segments=[Segment.from_dict(s) for s in d["segments"]],
            language=d.get("language"),
            duration=d.get("duration", 0.0),
        )


@dataclass(slots=True)
class Cue:
    """一則要輸出的字幕。primary 是主要顯示的文字，secondary 是雙語時的另一行。"""

    start: float
    end: float
    primary: str
    secondary: str | None = None
    speaker: str | None = None
