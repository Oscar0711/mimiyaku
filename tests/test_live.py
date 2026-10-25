import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from omni2.live.segmenter import Segmenter  # noqa: E402
from omni2.vad.presets import VadPreset  # noqa: E402
from omni2.vad.stream import FRAME  # noqa: E402

FRAME_S = FRAME / 16000  # 32 毫秒


class FakeVad:
    """依照給定的機率序列輸出，每 512 個取樣一格。"""

    def __init__(self, probs):
        self.probs = list(probs)
        self.pending = np.zeros(0, np.float32)
        self.i = 0

    def __call__(self, audio):
        self.pending = np.concatenate([self.pending, audio])
        out = []
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            p = self.probs[self.i] if self.i < len(self.probs) else 0.0
            self.i += 1
            out.append((frame, p))
        return out


PRESET = VadPreset(threshold=0.5, end_threshold=0.35, min_silence_ms=320, min_speech_ms=96, speech_pad_ms=64)


def run(probs, **kw):
    seg = Segmenter(PRESET, vad=FakeVad(probs), **kw)
    events = []
    audio = np.zeros(len(probs) * FRAME, np.float32)
    for i in range(0, len(audio), 1600):
        events += seg.feed(audio[i:i + 1600])
    return events + seg.flush()


def test_utterance_ends_after_silence():
    probs = [0.0] * 10 + [0.9] * 40 + [0.0] * 20 + [0.9] * 30 + [0.0] * 20
    finals = [e for e in run(probs, partial_ms=10_000) if e.final]
    assert len(finals) == 2
    assert finals[0].start == pytest.approx((10 - 2) * FRAME_S)  # 往前補 64 毫秒（2 格）
    assert finals[1].id == 1


def test_partials_while_speaking():
    probs = [0.9] * 60 + [0.0] * 20
    events = run(probs, partial_ms=320)
    partials = [e for e in events if not e.final]
    assert len(partials) >= 4
    assert all(len(b.audio) > len(a.audio) for a, b in zip(partials, partials[1:]))  # 草稿越來越長
    assert sum(e.final for e in events) == 1


def test_long_utterance_is_cut_at_breath():
    # 6 秒不停講話，中間 3.5 秒處有一小段機率低的「換氣」
    probs = [0.9] * 110 + [0.4] * 3 + [0.9] * 75 + [0.0] * 20
    finals = [e for e in run(probs, partial_ms=10_000, max_ms=4000) if e.final]
    assert len(finals) == 2
    assert len(finals[0].audio) // FRAME == pytest.approx(113, abs=3)


def test_short_noise_is_ignored():
    probs = [0.0] * 10 + [0.9] * 2 + [0.0] * 30
    assert [e for e in run(probs) if e.final] == []


class StreamClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def stream(self, messages, temperature=0.2, max_tokens=64):
        self.calls.append(messages)
        yield from self.reply


def test_live_translator_streams_and_caches():
    from omni2.live.translator import LiveTranslator
    from omni2.translate.glossary import Glossary, Term

    client = StreamClient(["大家", "晚安", "！\n（這是說明）"])
    t = LiveTranslator(client, "zh-TW", lambda lang: Glossary([Term("配信", "直播")]), opencc="s2twp")
    events = []
    final = {"id": "sys-0", "text": "こんばんは、配信です", "lang": "ja"}
    assert t.translate(final, events.append) == "大家晚安！"
    texts = [e["text"] for e in events if e["type"] == "translation"]
    assert texts[:2] == ["大家", "大家晚安"] and texts[-1] == "大家晚安！"
    assert any(e["type"] == "first_token" for e in events)
    assert "配信 → 直播" in client.calls[0][1]["content"]
    t.translate({"id": "sys-1", "text": "こんばんは、配信です", "lang": "ja"}, events.append)
    assert len(client.calls) == 1  # 第二次用快取
    assert "原：こんばんは、配信です" not in client.calls[0][1]["content"]


def test_live_translator_skips_same_language():
    from omni2.live.translator import LiveTranslator
    from omni2.translate.glossary import Glossary

    client = StreamClient([])
    t = LiveTranslator(client, "zh-TW", lambda lang: Glossary())
    assert t.translate({"id": "a", "text": "这个软件", "lang": "zh"}, lambda e: None) == "這個軟體"
    assert client.calls == []


def test_session_record_export(tmp_path, monkeypatch):
    from omni2.core import paths
    from omni2.live import record

    monkeypatch.setenv(paths.DATA_ENV, str(tmp_path))
    log = record.SessionLog({"tgt_lang": "zh-TW", "labels": {"mic": "我", "sys": "對方"}})
    log.add({"id": "sys-0", "src": "sys", "start": 1.0, "end": 2.0, "text": "Hello", "lang": "en",
             "translation": "你好"})
    log.add({"id": "mic-0", "src": "mic", "start": 2.5, "end": 3.0, "text": "Hi", "lang": "en",
             "translation": "嗨"})
    assert record.list_sessions() == [log.path]
    files = record.export_session(log.path, tmp_path, ["srt", "txt"])
    srt = files[0].read_text(encoding="utf-8")
    assert "00:00:01,000 -->" in srt and "你好" in srt
    txt = files[1].read_text(encoding="utf-8")
    assert "講者 A：Hello" in txt and "嗨" in txt
    assert "2 句" in record.describe(log.path)


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_overlay_fits_as_many_entries_as_the_window_allows(app):
    from PySide6.QtCore import Qt

    from omni2.config.settings import LiveSettings
    from omni2.ui import overlay as ov

    settings = LiveSettings(font_size=20)
    overlay = ov.SubtitleOverlay(settings)
    overlay.show()  # 還沒顯示的視窗，Qt 會把 resize 事件延到顯示時才送
    overlay.resize(800, 120)
    for i in range(6):
        overlay.set_text(f"e{i}", source=f"原文 {i}", translation=f"譯文 {i}")
    few = overlay.visible_ids()
    assert few and few[-1] == "e5"  # 最新的一定在、而且在最下面
    overlay.resize(800, 600)
    more = overlay.visible_ids()
    assert len(more) > len(few) and more[-1] == "e5"
    assert more == overlay.order[-len(more):]  # 顯示的是連續的最近幾句
    overlay.apply(LiveSettings(font_size=48))  # 字變大，放得下的句數變少
    assert len(overlay.visible_ids()) < len(more)
    overlay.resize(800, 30)  # 連一句都放不下時，最新一句照樣顯示
    assert overlay.visible_ids() == ["e5"]

    overlay.set_text("p", source="講到一半", partial=True)
    overlay.apply(settings)  # 重新套用樣式後，草稿仍是斜體
    assert overlay.entries["p"].source.font().italic()

    for i in range(ov.HISTORY + 5):
        overlay.set_text(f"x{i}", translation="…")
    assert len(overlay.order) == ov.HISTORY

    overlay.apply(LiveSettings(click_through=True))
    assert overlay.windowFlags() & Qt.WindowType.WindowTransparentForInput
    overlay.clear()
    assert overlay.order == [] and overlay.hint.isVisibleTo(overlay)


def test_live_page_handles_events(app, monkeypatch):
    from omni2.config.settings import Settings
    from omni2.ui import live_page as lp
    from omni2.ui.overlay import SubtitleOverlay

    monkeypatch.setattr(lp.LivePage, "_refresh_devices", lambda self: None)
    monkeypatch.setattr(lp, "save_settings", lambda s: None)
    settings = Settings()
    page = lp.LivePage(settings, SubtitleOverlay(settings.live))
    page.labels = {"sys": "對方"}
    page._on_event({"type": "partial", "id": "sys-0", "src": "sys", "start": 1.0, "end": 2.0, "text": "こん"})
    page._on_event({"type": "final", "id": "sys-0", "src": "sys", "start": 1.0, "end": 2.5,
                    "text": "こんにちは", "lat_text": 0.8})
    page._on_event({"type": "translation", "id": "sys-0", "text": "你好", "done": True})
    page._on_event({"type": "latency", "id": "sys-0", "lat_first": 0.9})
    assert page.table.rowCount() == 1
    assert [page.table.item(0, c).text() for c in range(4)] == ["00:00:01", "對方", "こんにちは", "你好"]
    assert page.overlay.entries["sys-0"].translation.text() == "你好"
    assert "0.8" in page.latency.text() and "0.9" in page.latency.text()
