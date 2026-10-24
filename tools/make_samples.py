"""產生測試樣本：用 edge-tts 合成日、英、韓口語短片，句間插入停頓。

每個樣本另外輸出 <name>.ref.json，記錄每句的正確文字和時間，可拿來算辨識錯誤率。

    uv run --with edge-tts python tools/make_samples.py [輸出資料夾]
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from omni2.audio.decode import SAMPLE_RATE, decode_to_wav, read_wav

SAMPLES: dict[str, list[tuple[str, str]]] = {
    "stream_ja": [("ja-JP-NanamiNeural", t) for t in [
        "みなさん、こんばんは！今日も配信に来てくれてありがとうございます。",
        "今日は新しいゲームをやっていこうと思います。",
        "実は昨日、夜遅くまで練習してたんですよ。",
        "でも全然うまくならなくて、ちょっと落ち込みました。",
        "あ、コメントありがとう！がんばってって言ってくれてうれしいです。",
        "このゲーム、ボスがめちゃくちゃ強いらしいんですよね。",
        "前回の配信では三回も負けちゃったので、今日こそリベンジしたいです。",
        "それと、来週の土曜日に記念ライブがあります。",
        "チケットはまだ少し残っているので、気になる人はぜひチェックしてください。",
        "じゃあ、そろそろ始めましょうか。",
        "よし、いくぞ！",
        "うわ、いきなり敵が出てきた！ちょっと待って待って！",
    ]],
    "talk_en": [("en-US-AndrewNeural", t) for t in [
        "Hi everyone, and welcome back to the channel.",
        "Today I want to talk about something that changed the way I work: taking real breaks.",
        "For years, I thought that working longer meant getting more done.",
        "But the research says otherwise. After about ninety minutes, our focus drops sharply.",
        "So last spring, I tried an experiment.",
        "Every hour and a half, I stood up, went outside, and walked for ten minutes without my phone.",
        "The first week felt strange. I kept worrying that I was falling behind.",
        "By the third week, though, I was finishing my tasks earlier than before.",
        "My colleague Sarah tried it too, and she said her afternoon headaches disappeared.",
        "Of course, this won't work for everyone.",
        "But if you're feeling stuck, give it a try for two weeks and let me know how it goes in the comments.",
    ]],
    "vlog_ko": [("ko-KR-SunHiNeural", t) for t in [
        "안녕하세요, 여러분! 오늘은 서울에서 제일 좋아하는 카페를 소개해 드릴게요.",
        "이 카페는 성수동에 있는데요, 분위기가 정말 좋아요.",
        "저는 보통 주말 아침에 여기 와서 책을 읽어요.",
        "오늘은 딸기 라떼를 주문했어요. 진짜 맛있어요!",
        "사장님이 직접 빵도 구우셔서 냄새가 너무 좋아요.",
        "아, 그리고 이 카페는 강아지도 같이 들어올 수 있어요.",
        "그래서 주말에는 강아지랑 같이 오는 손님이 많아요.",
        "다만 자리가 많지 않아서 일찍 오시는 게 좋아요.",
        "다음 영상에서는 근처에 있는 맛집도 소개해 드릴게요.",
        "오늘 영상이 도움이 되셨다면 구독과 좋아요 부탁드려요!",
    ]],
    # 兩位講者輪流說話，之後測講者分離用
    "meeting_ja": [
        ("ja-JP-KeitaNeural", "では、来月のイベントについて打ち合わせを始めましょう。"),
        ("ja-JP-NanamiNeural", "はい。会場はもう予約できました。駅から歩いて五分のホールです。"),
        ("ja-JP-KeitaNeural", "ありがとうございます。参加者は何人くらいの予定ですか？"),
        ("ja-JP-NanamiNeural", "今のところ、申し込みは百二十人です。"),
        ("ja-JP-KeitaNeural", "思ったより多いですね。椅子は足りますか？"),
        ("ja-JP-NanamiNeural", "ホールに百五十席あるので、大丈夫だと思います。"),
        ("ja-JP-KeitaNeural", "わかりました。では、受付は誰が担当しますか？"),
        ("ja-JP-NanamiNeural", "私と田中さんでやります。"),
        ("ja-JP-KeitaNeural", "よろしくお願いします。次回は来週の水曜日に確認しましょう。"),
    ],
    # 三位女性快速聊天、很多短回應：講者分離的困難樣本（後兩位是會講日文的多語言語音）
    "chat_ja": [
        ("ja-JP-NanamiNeural", "ねえ、昨日のライブ見た？"),
        ("en-US-AvaMultilingualNeural", "見た見た！"),
        ("ja-JP-NanamiNeural", "最後の曲、すごくよかったよね。"),
        ("en-US-EmmaMultilingualNeural", "え、私見てない。何の曲だったの？"),
        ("en-US-AvaMultilingualNeural", "新曲だよ。まだ配信されてないやつ。"),
        ("en-US-EmmaMultilingualNeural", "えー、うそ。"),
        ("ja-JP-NanamiNeural", "アーカイブ残ってるから、今日中に見たほうがいいよ。"),
        ("en-US-EmmaMultilingualNeural", "わかった、帰ったら見る。"),
        ("en-US-AvaMultilingualNeural", "あとね、グッズも出るんだって。"),
        ("ja-JP-NanamiNeural", "そうそう。Tシャツとアクスタ。"),
        ("en-US-EmmaMultilingualNeural", "アクスタほしい！"),
        ("en-US-AvaMultilingualNeural", "来週の金曜日から予約開始らしいよ。"),
        ("ja-JP-NanamiNeural", "じゃあ、みんなで一緒に買おうか。"),
        ("en-US-EmmaMultilingualNeural", "うん、そうしよう。"),
        ("en-US-AvaMultilingualNeural", "決まりだね。"),
    ],
    # 兩位男主持人＋一位女來賓、停頓很短
    "podcast_en": [
        ("en-US-GuyNeural", "Welcome back to the show. Today we've got a special guest."),
        ("en-US-JennyNeural", "Thanks for having me."),
        ("en-US-ChristopherNeural", "Great to have you here. So, how did you get started?"),
        ("en-US-JennyNeural", "Honestly, by accident. I just started posting videos during the lockdown."),
        ("en-US-GuyNeural", "Right."),
        ("en-US-JennyNeural", "And people kept watching, so I kept going."),
        ("en-US-ChristopherNeural", "That's amazing. Did you ever expect it to grow this big?"),
        ("en-US-JennyNeural", "Not at all."),
        ("en-US-GuyNeural", "What was the hardest part?"),
        ("en-US-JennyNeural", "Probably staying consistent. Some weeks I had no ideas at all."),
        ("en-US-ChristopherNeural", "I can relate to that."),
        ("en-US-GuyNeural", "Yeah, same here."),
        ("en-US-ChristopherNeural", "Okay, one last question before we wrap up."),
        ("en-US-JennyNeural", "Sure, go ahead."),
        ("en-US-ChristopherNeural", "What advice would you give someone just starting out?"),
        ("en-US-JennyNeural", "Just start. Your first videos will be bad, and that's fine."),
        ("en-US-GuyNeural", "Great advice. Thanks so much for coming."),
    ],
}

# 句間停頓（秒）；對話樣本接得比較快
PAUSES = {"chat_ja": (0.15, 0.5), "podcast_en": (0.15, 0.5)}

_K, _N, _A = "ja-JP-KeitaNeural", "ja-JP-NanamiNeural", "en-US-AvaMultilingualNeural"
# 依時間軸混音的樣本：（語音, 文字, 和上一句結尾的間隔秒數）。間隔是負的代表重疊（插話、附和）。
# 每句前後的靜音會先切掉，間隔才是真正的「接話速度」。
TIMELINE_SAMPLES: dict[str, list[tuple[str, str, float]]] = {
    # 三人快速接話、很多一兩個字的短回應
    "quick_ja": [
        (_K, "今日の会議、何時からだっけ？", 0.0),
        (_N, "三時から。", 0.10),
        (_K, "え、もう？", 0.08),
        (_A, "うん、あと十分しかないよ。", 0.10),
        (_K, "やばい、資料まだ印刷してない。", 0.12),
        (_N, "私やっとくよ。", 0.06),
        (_K, "ほんと？助かる。", 0.10),
        (_A, "会議室はどこ？", 0.15),
        (_N, "二階の大きい部屋。", 0.08),
        (_A, "了解。", 0.05),
        (_K, "じゃあ先に行ってて。", 0.10),
        (_N, "はーい。", 0.08),
        (_A, "後でね。", 0.10),
    ],
    # 對方還沒講完就附和或插話
    "overlap_ja": [
        (_K, "昨日さ、駅前で偶然、高校の時の先生に会ったんだよね。", 0.0),
        (_N, "へえ！", -0.6),
        (_K, "全然変わってなくてびっくりした。", 0.10),
        (_N, "うんうん。", -0.5),
        (_N, "それで、何か話したの？", 0.20),
        (_K, "ちょっとだけね。今度、同窓会やるらしいよ。", 0.10),
        (_A, "え、私も行きたい！", -0.4),
        (_K, "もちろん、みんなに連絡するつもり。", 0.10),
        (_N, "いつやるの？", 0.10),
        (_K, "来月の第二土曜日だって。", 0.05),
        (_A, "あ、その日空いてる。", -0.3),
        (_N, "私も大丈夫。", 0.10),
    ],
}


def _trim(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    loud = np.nonzero(np.abs(audio) > threshold)[0]
    return audio[loud[0]: loud[-1] + 1] if len(loud) else audio


async def build_timeline(name: str, lines: list[tuple[str, str, float]], out_dir: Path) -> Path:
    clips = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (voice, text, _) in enumerate(lines):
            mp3, wav = Path(tmp) / f"{i}.mp3", Path(tmp) / f"{i}.wav"
            await _speak(voice, text, mp3)
            decode_to_wav(mp3, wav)
            clips.append(_trim(read_wav(wav)))
    refs, t, end = [], 0.8, 0.8
    placed = []
    for (voice, text, gap), clip in zip(lines, clips):
        start = max(0.0, end + gap) if refs else t
        placed.append((int(start * SAMPLE_RATE), clip))
        end = start + len(clip) / SAMPLE_RATE
        refs.append({"start": round(start, 2), "end": round(end, 2), "text": text, "speaker": voice})
    total = max(s + len(c) for s, c in placed) + int(0.8 * SAMPLE_RATE)
    mix = np.zeros(total, np.float32)
    for s, c in placed:
        mix[s: s + len(c)] += c
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    path = out_dir / f"{name}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    (out_dir / f"{name}.ref.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


async def _speak(voice: str, text: str, out: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice).save(str(out))


async def build(name: str, lines: list[tuple[str, str]], out_dir: Path, rng: random.Random) -> Path:
    chunks: list[np.ndarray] = [np.zeros(int(SAMPLE_RATE * 0.8), np.float32)]
    refs = []
    t = len(chunks[0]) / SAMPLE_RATE
    with tempfile.TemporaryDirectory() as tmp:
        for i, (voice, text) in enumerate(lines):
            mp3 = Path(tmp) / f"{i}.mp3"
            wav = Path(tmp) / f"{i}.wav"
            await _speak(voice, text, mp3)
            decode_to_wav(mp3, wav)
            audio = read_wav(wav)
            refs.append({"start": round(t, 2), "end": round(t + len(audio) / SAMPLE_RATE, 2),
                         "text": text, "speaker": voice})
            chunks.append(audio)
            t += len(audio) / SAMPLE_RATE
            pause = np.zeros(int(SAMPLE_RATE * rng.uniform(*PAUSES.get(name, (0.5, 1.3)))), np.float32)
            chunks.append(pause)
            t += len(pause) / SAMPLE_RATE
    pcm = (np.clip(np.concatenate(chunks), -1, 1) * 32767).astype(np.int16)
    path = out_dir / f"{name}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    (out_dir / f"{name}.ref.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def synth_music(seconds: float, seed: int = 3) -> np.ndarray:
    """合成背景音樂：和弦（帶泛音）＋大鼓、小鼓、鈸，給人聲分離測試用。"""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    out = np.zeros(n, np.float32)
    beat = 60 / 110
    chords = [[220.0, 277.2, 329.6], [196.0, 246.9, 293.7], [174.6, 220.0, 261.6], [196.0, 246.9, 311.1]]
    bar = 4 * beat
    for k in range(int(seconds / bar) + 1):
        a, b = int(k * bar * SAMPLE_RATE), min(n, int((k + 1) * bar * SAMPLE_RATE))
        tt = t[a:b] - t[a] if a < n else t[:0]
        for f in chords[k % 4]:
            for h, amp in ((1, 1.0), (2, 0.5), (3, 0.25)):
                out[a:b] += amp * 0.06 * np.sin(2 * np.pi * f * h * tt) * np.exp(-tt / 3)
    for k in range(int(seconds / beat) + 1):
        a = int(k * beat * SAMPLE_RATE)
        for length, sound in (
            (0.25, lambda tt: np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 18) * 0.6),  # 大鼓
            (0.15, lambda tt: rng.normal(size=len(tt)) * np.exp(-tt * 25) * (0.25 if k % 2 else 0.05)),  # 小鼓
        ):
            b = min(n, a + int(length * SAMPLE_RATE))
            if a < b:
                out[a:b] += sound(np.arange(b - a) / SAMPLE_RATE).astype(np.float32)
        for half in (0, 0.5):  # 鈸
            a2 = int((k + half) * beat * SAMPLE_RATE)
            b2 = min(n, a2 + int(0.05 * SAMPLE_RATE))
            if a2 < b2:
                out[a2:b2] += (rng.normal(size=b2 - a2) * 0.08).astype(np.float32)
    return out


def add_bgm(src: Path, dst: Path) -> Path:
    """把語音和合成音樂以接近 1:1 的音量混在一起（相當吵）。"""
    speech = read_wav(src)
    music = synth_music(len(speech) / SAMPLE_RATE)
    music *= np.sqrt(np.mean(speech**2) / max(np.mean(music**2), 1e-9))
    mix = np.clip(speech + music, -1, 1)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    dst.with_suffix(".ref.json").write_text(src.with_suffix(".ref.json").read_text(encoding="utf-8"),
                                            encoding="utf-8")
    return dst


async def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    for name, lines in SAMPLES.items():
        if not (out_dir / f"{name}.wav").exists():
            print(await build(name, lines, out_dir, rng))
    print(add_bgm(out_dir / "stream_ja.wav", out_dir / "stream_ja_bgm.wav"))
    for name, lines in TIMELINE_SAMPLES.items():
        if not (out_dir / f"{name}.wav").exists():
            print(await build_timeline(name, lines, out_dir))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1] if len(sys.argv) > 1 else "samples")))
