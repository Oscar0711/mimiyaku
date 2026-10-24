"""Google Cloud Speech-to-Text v2（預設 Chirp 3）。

逐段（60 秒以內）送同步辨識，最多 4 段同時進行，要求回傳逐詞時間。
金鑰是服務帳戶金鑰 JSON 的路徑；沒有設定時使用 Application Default Credentials。
※ 尚未用真實金鑰實測，只用模擬的回應測過。
"""

from __future__ import annotations

import numpy as np

from omni2.audio.decode import SAMPLE_RATE
from omni2.audio.encode import wav_bytes
from omni2.core import lang
from omni2.core.secrets import get_secret
from omni2.core.types import Segment, Transcript
from omni2.post.text import join_texts
from omni2.stt.base import ProgressFn, SttEngine, TranscribeOptions
from omni2.stt.cloud_clips import transcribe_clips
from omni2.stt.words import attach_punctuation, spread_words

LANGUAGE_CODES = {"ja": "ja-JP", "en": "en-US", "ko": "ko-KR", "zh": "cmn-Hans-CN",
                  "zh-TW": "cmn-Hant-TW", "zh-CN": "cmn-Hans-CN", "yue": "yue-Hant-HK"}


class GoogleEngine(SttEngine):
    name = "google"

    def load(self) -> None:
        from google.cloud.speech_v2 import SpeechClient

        c = self.settings.cloud
        if not c.google_project:
            raise RuntimeError("還沒設定 Google Cloud 的專案 ID：請到「設定」分頁填入")
        credentials = None
        key_path = get_secret("google")
        if key_path:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(key_path)
        options = None if c.google_location == "global" else {
            "api_endpoint": f"{c.google_location}-speech.googleapis.com"}
        self.client = SpeechClient(credentials=credentials, client_options=options)
        self.recognizer = f"projects/{c.google_project}/locations/{c.google_location}/recognizers/_"
        self.model = c.google_model

    def transcribe(self, audio: np.ndarray, opts: TranscribeOptions,
                   progress: ProgressFn | None = None) -> Transcript:
        from google.cloud.speech_v2.types import cloud_speech as cs

        code = (LANGUAGE_CODES.get(opts.language or "") or LANGUAGE_CODES.get(lang.base(opts.language))
                or "auto")
        config = cs.RecognitionConfig(
            auto_decoding_config=cs.AutoDetectDecodingConfig(),
            language_codes=[code],
            model=self.model,
            features=cs.RecognitionFeatures(enable_word_time_offsets=True, enable_automatic_punctuation=True),
        )
        detected: list[str] = []

        def one(clip: np.ndarray, start: float, end: float) -> Segment | None:
            request = cs.RecognizeRequest(recognizer=self.recognizer, config=config, content=wav_bytes(clip))
            response = self.client.recognize(request=request)
            texts, items = [], []
            for result in response.results:
                if not result.alternatives:
                    continue
                alt = result.alternatives[0]
                texts.append(alt.transcript)
                items += [(w.word, start + w.start_offset.total_seconds(), start + w.end_offset.total_seconds())
                          for w in alt.words]
                if result.language_code:
                    detected.append(lang.base(result.language_code))
            text = join_texts(texts)
            if not text:
                return None
            words = attach_punctuation(text, items) if items else spread_words(text, start, end)
            return Segment(start=start, end=end, text=text, words=words)

        segments = transcribe_clips(audio, opts, one, progress)
        language = opts.language or (max(set(detected), key=detected.count) if detected else None)
        return Transcript(segments=segments, language=language, duration=len(audio) / SAMPLE_RATE)
