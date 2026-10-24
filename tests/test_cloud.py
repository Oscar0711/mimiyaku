"""雲端引擎：全部用模擬的回應測試，不會連線、不需要金鑰。"""

import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omni2.config.settings import SttSettings
from omni2.core import lang, secrets
from omni2.stt.base import TranscribeOptions
from omni2.stt.words import spread_words
from omni2.vad.presets import VadPreset

SR = 16000

FUNASR_RESULT = {
    "transcripts": [{
        "channel_id": 0,
        "sentences": [
            {"begin_time": 1200, "end_time": 3000, "text": "こんにちは、皆さん。", "words": [
                {"begin_time": 1200, "end_time": 2000, "text": "こんにちは", "punctuation": "、"},
                {"begin_time": 2100, "end_time": 3000, "text": "皆さん", "punctuation": "。"}]},
            {"begin_time": 100, "end_time": 900, "text": "はい。", "words": [
                {"begin_time": 100, "end_time": 900, "text": "はい", "punctuation": "。"}]},
        ],
    }],
}


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, name):
        return self.store.get((service, name))

    def set_password(self, service, name, value):
        self.store[(service, name)] = value

    def delete_password(self, service, name):
        del self.store[(service, name)]


@pytest.fixture
def keyring(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(secrets, "_keyring", lambda: fake)
    for env in secrets.ENV_VARS.values():
        monkeypatch.delenv(env, raising=False)
    return fake


def test_secrets_keyring_then_env(keyring, monkeypatch):
    assert secrets.get_secret("dashscope") is None and secrets.secret_source("dashscope") is None
    monkeypatch.setenv("DASHSCOPE_API_KEY", "from-env")
    assert secrets.get_secret("dashscope") == "from-env" and secrets.secret_source("dashscope") == "env"
    secrets.set_secret("dashscope", "from-keyring")
    assert secrets.get_secret("dashscope") == "from-keyring" and secrets.secret_source("dashscope") == "keyring"
    secrets.set_secret("dashscope", "")
    assert secrets.get_secret("dashscope") == "from-env"
    with pytest.raises(secrets.MissingSecret, match="OpenAI"):
        secrets.require_secret("openai", "OpenAI")


def test_guess_language():
    assert lang.guess_from_text("今日はいい天気ですね") == "ja"
    assert lang.guess_from_text("오늘은 날씨가 좋네요") == "ko"
    assert lang.guess_from_text("今天天氣很好") == "zh"
    assert lang.guess_from_text("Nice weather today") == "en"
    assert lang.guess_from_text("") is None


def test_spread_words_covers_span_in_order():
    words = spread_words("Hello there, world.", 1.0, 4.0)
    assert "".join(w.text for w in words).strip() == "Hello there, world."
    assert words[0].start == 1.0 and words[-1].end == pytest.approx(4.0)
    assert all(a.end == pytest.approx(b.start) for a, b in zip(words, words[1:]))


def test_funasr_parse_result_sorts_and_attaches_punctuation():
    from omni2.stt.funasr_engine import parse_result

    segs = parse_result(FUNASR_RESULT)
    assert [s.text for s in segs] == ["はい。", "こんにちは、皆さん。"]
    assert [w.text for w in segs[1].words] == ["こんにちは、", "皆さん。"]
    assert segs[1].words[1].start == 2.1


def test_funasr_engine_flow(keyring, monkeypatch, tmp_path):
    from dashscope.audio.asr import Transcription
    from dashscope.utils.oss_utils import OssUtils

    from omni2.stt import funasr_engine as fe

    keyring.set_password(secrets.SERVICE, "dashscope", "sk-test")
    uploads, calls = [], {}

    def fake_upload(model, file_path, api_key, upload_certificate=None):
        uploads.append((model, api_key, upload_certificate, Path(file_path).name))
        return f"oss://dashscope-instant/abc/{Path(file_path).name}", {"cert": 1}

    def fake_async_call(model, file_urls, api_key, **kwargs):
        calls["submit"] = (model, file_urls, kwargs)
        return SimpleNamespace(status_code=200, output={"task_id": "t1", "task_status": "PENDING"})

    states = iter(["RUNNING", "SUCCEEDED"])

    def fake_fetch(task, api_key):
        # 實際的服務：回傳的是 https 網址、順序和送出時不同
        names = [u[3] for u in uploads]
        results = [{"file_url": f"https://dashscope-instant.oss.aliyuncs.com/abc/{n}?Expires=1",
                    "subtask_status": "SUCCEEDED", "transcription_url": f"https://result/{n}"}
                   for n in reversed(names)]
        return SimpleNamespace(status_code=200, output={"task_status": next(states), "results": results})

    def fake_get(url, timeout):
        # 第一個檔案的結果只有一句，第二個有兩句，用來確認沒有對錯檔案
        first = url.endswith(uploads[0][3])
        data = {"transcripts": [{"sentences": FUNASR_RESULT["transcripts"][0]["sentences"][: 1 if first else 2]}]}
        return SimpleNamespace(json=lambda: data)

    monkeypatch.setattr(OssUtils, "upload", staticmethod(fake_upload))
    monkeypatch.setattr(Transcription, "async_call", staticmethod(fake_async_call))
    monkeypatch.setattr(Transcription, "fetch", staticmethod(fake_fetch))
    monkeypatch.setattr(fe.time, "sleep", lambda s: None)
    monkeypatch.setattr(fe.httpx, "get", fake_get)
    monkeypatch.setattr(fe, "encode_opus", lambda audio, dst: (dst.write_bytes(b"ogg"), dst)[1])

    engine = fe.FunAsrEngine(SttSettings(engine="funasr"))
    engine.load()
    progress = []
    out = engine.transcribe_many([np.zeros(SR * 5, np.float32)] * 2, TranscribeOptions(language="ja"),
                                 lambda i, p: progress.append((i, p)))
    assert [len(t.segments) for t in out] == [1, 2] and out[0].language == "ja"
    assert uploads[0][:3] == ("fun-asr", "sk-test", None) and uploads[1][2] == {"cert": 1}  # 第二次沿用憑證
    assert uploads[0][3] != uploads[1][3]  # 每個檔案的檔名都不同
    model, urls, kwargs = calls["submit"]
    assert len(urls) == 2 and all(u.startswith("oss://") for u in urls)
    assert kwargs["language_hints"] == ["ja"]
    assert kwargs["headers"] == {"X-DashScope-OssResourceResolve": "enable"}
    assert (1, 1.0) in progress


def test_funasr_requires_key(keyring):
    from omni2.stt.funasr_engine import FunAsrEngine

    with pytest.raises(secrets.MissingSecret):
        FunAsrEngine(SttSettings(engine="funasr")).load()


def _speech(seconds_on=(1.0, 3.0), total=5.0):
    audio = np.zeros(int(SR * total), np.float32)
    a, b = (int(x * SR) for x in seconds_on)
    audio[a:b] = 0.3 * np.sin(np.arange(b - a) * 2 * np.pi * 220 / SR)
    return audio


@pytest.fixture
def one_clip(monkeypatch):
    """不用真的跑 VAD：整段當成一個語音區段（1～3 秒）。"""
    from omni2.stt import cloud_clips

    monkeypatch.setattr(cloud_clips, "make_clips", lambda audio, vad: [{"start": 1.0, "end": 3.0}])


def test_openai_whisper1_uses_word_timestamps(keyring, one_clip):
    from omni2.stt.openai_engine import OpenAiEngine

    keyring.set_password(secrets.SERVICE, "openai", "sk-test")
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(text="Hello, world.", language="english",
                               words=[SimpleNamespace(word="Hello", start=0.1, end=0.5),
                                      SimpleNamespace(word="world", start=0.6, end=1.2)])

    settings = SttSettings(engine="openai")
    settings.cloud.openai_model = "whisper-1"
    engine = OpenAiEngine(settings)
    engine.load()
    engine.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    tr = engine.transcribe(_speech(), TranscribeOptions(hotwords=["Sarah"]))
    assert tr.language == "en"
    assert [(w.text, w.start) for w in tr.segments[0].words] == [("Hello,", 1.1), (" world.", 1.6)]
    assert sent["prompt"] == "Sarah" and sent["timestamp_granularities"] == ["word"]


def test_openai_gpt4o_spreads_words(keyring, one_clip):
    from omni2.stt.openai_engine import OpenAiEngine

    keyring.set_password(secrets.SERVICE, "openai", "sk-test")
    engine = OpenAiEngine(SttSettings(engine="openai"))
    engine.load()
    engine.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(text="今日はいい天気ですね。"))))
    tr = engine.transcribe(_speech(), TranscribeOptions())
    seg = tr.segments[0]
    assert (seg.start, seg.end, tr.language) == (1.0, 3.0, "ja")
    assert seg.words[0].start == 1.0 and seg.words[-1].end == pytest.approx(3.0)


def test_google_engine_word_offsets(keyring, one_clip, monkeypatch):
    from omni2.stt import google_engine as ge

    settings = SttSettings(engine="google")
    settings.cloud.google_project = "proj"
    monkeypatch.setattr("google.cloud.speech_v2.SpeechClient", lambda **kw: None)
    engine = ge.GoogleEngine(settings)
    engine.load()
    assert engine.recognizer == "projects/proj/locations/us/recognizers/_"
    td = datetime.timedelta
    word = lambda w, a, b: SimpleNamespace(word=w, start_offset=td(seconds=a), end_offset=td(seconds=b))
    result = SimpleNamespace(language_code="ja-jp", alternatives=[SimpleNamespace(
        transcript="こんにちは。", words=[word("こんにちは", 0.2, 0.9)])])
    seen = {}

    def recognize(request):
        seen["request"] = request
        return SimpleNamespace(results=[result])

    engine.client = SimpleNamespace(recognize=recognize)
    tr = engine.transcribe(_speech(), TranscribeOptions(language="ja"))
    assert tr.segments[0].words[0].text == "こんにちは。"
    assert tr.segments[0].words[0].start == pytest.approx(1.2)
    assert list(seen["request"].config.language_codes) == ["ja-JP"]


def test_google_requires_project(keyring):
    from omni2.stt.google_engine import GoogleEngine

    with pytest.raises(RuntimeError, match="專案 ID"):
        GoogleEngine(SttSettings(engine="google")).load()


def test_encode_opus_produces_ogg(tmp_path):
    from omni2.audio.encode import encode_opus

    out = encode_opus(_speech(), tmp_path / "a.ogg")
    assert out.read_bytes()[:4] == b"OggS"


def test_funasr_tidy_text():
    from omni2.stt.funasr_engine import tidy_text

    assert tidy_text("このゲーム ボスがめちゃくちゃ強い", "ja") == "このゲームボスがめちゃくちゃ強い"
    assert tidy_text("じゃあ そろそろ始めましょうか？", "ja") == "じゃあそろそろ始めましょうか？"
    assert tidy_text("today i want to say i'm fine.", "en") == "Today I want to say I'm fine."
    assert tidy_text("다만 자리가 많지 않아서", "ko") == "다만 자리가 많지 않아서"  # 韓文的空格不動
    assert tidy_text("用 Python 寫", "zh") == "用 Python 寫"  # 中英之間的空格保留


def test_funasr_parse_result_tidies_english():
    from omni2.stt.funasr_engine import parse_result

    data = {"transcripts": [{"sentences": [{"begin_time": 0, "end_time": 1000, "text": "the first week i felt strange.",
                                            "words": [{"begin_time": 0, "end_time": 300, "text": "the"},
                                                      {"begin_time": 300, "end_time": 600, "text": "first"}]}]}]}
    seg = parse_result(data)[0]
    assert seg.text == "The first week I felt strange."
    assert seg.words[0].text == "The"
