import json

import numpy as np

from omni2.core.types import Segment, Word
from omni2.post.diarize import (
    Turn,
    absorb_small,
    assign_speakers,
    cluster,
    speech_units,
    speech_windows,
    windows_to_turns,
)
from omni2.translate.client import ChatResult
from omni2.translate.glossary import Glossary, Term
from omni2.translate.terms import chunk_lines, extract_terms


def _voice(seed, n):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=64)
    return [base + rng.normal(scale=0.15, size=64) for _ in range(n)]


def test_cluster_separates_two_voices_in_order_of_appearance():
    a, b = _voice(1, 3), _voice(2, 3)
    emb = np.stack([b[0], a[0], b[1], a[1], a[2], b[2]])
    assert cluster(emb, threshold=0.5) == [0, 1, 0, 1, 1, 0]


def test_cluster_with_known_speaker_count():
    a, b, c = _voice(1, 2), _voice(2, 2), _voice(3, 2)
    emb = np.stack(a + b + c)
    assert len(set(cluster(emb, threshold=0.5, num_speakers=2))) == 2


def test_speech_windows_cover_long_and_short_segments():
    segs = [Segment(0.0, 4.0, "a"), Segment(5.0, 6.0, "b"), Segment(7.0, 7.2, "c")]
    wins = speech_windows([(s.start, s.end) for s in segs])
    assert wins[0] == (0.0, 1.5) and wins[-2] == (2.5, 4.0)
    assert (5.0, 6.0) in wins  # 短句整句一個窗口
    assert all(s >= 5.0 or e <= 4.0 for s, e in wins)
    assert not any(s >= 7.0 for s, _ in wins)  # 太短的不算


def test_windows_to_turns_merges_same_speaker():
    wins = [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0), (2.25, 3.75)]
    turns = windows_to_turns(wins, [0, 0, 1, 1])
    assert [(t.speaker, round(t.start, 3), round(t.end, 3)) for t in turns] == [(0, 0.0, 1.875), (1, 1.875, 3.75)]


def _words(spec):
    return [Word(s, e, t) for s, e, t in spec]


def test_assign_speakers_splits_at_speaker_change():
    words = _words([(0.0, 0.5, "見た"), (0.5, 1.0, "見た！"), (1.6, 2.0, "最後"), (2.0, 2.6, "の曲、"),
                    (2.6, 3.4, "よかったね。")])
    seg = Segment(0.0, 3.4, "見た見た！最後の曲、よかったね。", words)
    turns = [Turn(0.0, 1.3, 7), Turn(1.3, 3.5, 3)]
    out = assign_speakers([seg], turns)
    assert [(s.text, s.speaker) for s in out] == [("見た見た！", "SPEAKER_00"), ("最後の曲、よかったね。", "SPEAKER_01")]
    assert out[0].start == 0.0 and out[-1].end == 3.4


def test_assign_speakers_smooths_short_flips():
    words = _words([(0.0, 1.0, "a"), (1.0, 1.2, " b"), (1.2, 2.5, " c")])
    turns = [Turn(0.0, 1.0, 0), Turn(1.0, 1.2, 1), Turn(1.2, 2.5, 0)]
    out = assign_speakers([Segment(0.0, 2.5, "a b c", words)], turns)
    assert [(s.text, s.speaker) for s in out] == [("a b c", "SPEAKER_00")]


def test_absorb_small_merges_noise_cluster():
    a, b = _voice(1, 10), _voice(2, 10)
    noise = _voice(1, 1)
    emb = np.stack(a + b + noise)
    labels = [0] * 10 + [1] * 10 + [2]
    assert absorb_small(labels, emb, 0.1) == [0] * 10 + [1] * 10 + [0]


def test_builtin_glossaries_load():
    ja = Glossary.builtin("streaming", "ja")
    assert any(t.source == "スパチャ" and t.target == "SC" for t in ja.terms)
    assert Glossary.builtin("streaming", "ko").matches("성수동에 있는데요")[0].target == "聖水洞"
    assert len(Glossary.builtin("streaming", "th")) == 0


def test_english_terms_match_whole_words_only():
    g = Glossary([Term("mod", "管理員"), Term("GG", "GG")])
    assert g.matches("ask a mod for help")
    assert not g.matches("this model is great")
    assert [t.source for t in g.matches("gg everyone")] == ["GG"]


def test_chunk_lines():
    assert chunk_lines(["aaaa", "bbbb", "cc"], 9) == ["aaaa\nbbbb", "cc"]
    assert chunk_lines(["x" * 20], 5) == ["x" * 20]


class TermsClient:
    def __init__(self):
        self.prompts = []

    def complete(self, req, *, temperature, max_tokens):
        user = req.messages[-1]["content"]
        self.prompts.append(user)
        if "合併成一段" in user:
            return ChatResult(json.dumps({"summary": "合併後的摘要"}, ensure_ascii=False), "stop")
        return ChatResult(json.dumps({"summary": "片段摘要", "terms": [
            {"source": "田中", "target": "田中", "type": "人名", "note": "同事"},
            {"source": "鈴木", "target": "鈴木", "type": "人名", "note": "模型自己編的"},
            {"source": "スパチャ", "target": "打賞", "type": "網路用語", "note": ""},
        ]}, ensure_ascii=False), "stop")


def test_extract_terms_filters_invented_and_known_terms():
    client = TermsClient()
    known = Glossary([Term("スパチャ", "SC")])
    result = extract_terms(client, ["田中さんです", "スパチャありがとう"], src_lang="ja", tgt_lang="zh-TW",
                           known=known)
    assert [t["source"] for t in result.terms] == ["田中"]
    assert result.summary == "片段摘要"
    assert "スパチャ → SC" in client.prompts[0]  # 已知詞連同譯名告訴模型


def test_extract_terms_merges_chunk_summaries():
    client = TermsClient()
    result = extract_terms(client, ["田中" * 10, "田中" * 10], src_lang="ja", tgt_lang="zh-TW", chunk_chars=25)
    assert result.summary == "合併後的摘要"
    assert len(client.prompts) == 3
    assert [t["source"] for t in result.terms] == ["田中"]


def test_cluster_known_count_ignores_outlier_windows():
    a, b, c = _voice(1, 12), _voice(2, 12), _voice(3, 12)
    outlier = [np.random.default_rng(9).normal(size=64)]  # 例如大喊、笑聲
    emb = np.stack(a + b + c + outlier)
    labels = cluster(emb, num_speakers=3)
    assert len(set(labels)) == 3
    assert len(set(labels[:12])) == len(set(labels[12:24])) == len(set(labels[24:36])) == 1


def test_speech_units_split_at_sentence_end_and_pause():
    words = _words([(0.0, 0.5, "三時"), (0.5, 0.9, "から。"), (1.0, 1.4, "え、"), (1.4, 1.8, "もう？"),
                    (2.3, 2.8, "うん")])
    units = speech_units([Segment(0.0, 2.8, "三時から。え、もう？うん", words)])
    assert units == [(0.0, 0.9), (1.0, 1.4), (1.4, 1.8), (2.3, 2.8)]


def test_snap_moves_boundary_to_sentence_end():
    # 聲紋把「か」判給了下一個人；切點應該挪回句尾「か。」之後
    words = _words([(0.0, 0.5, "買おう"), (0.5, 0.7, "か。"), (0.72, 1.2, "うん、"), (1.2, 2.0, "そうしよう。")])
    turns = [Turn(0.0, 0.5, 0), Turn(0.5, 2.0, 1)]
    out = assign_speakers([Segment(0.0, 2.0, "買おうか。うん、そうしよう。", words)], turns)
    assert [(s.text, s.speaker) for s in out] == [("買おうか。", "SPEAKER_00"), ("うん、そうしよう。", "SPEAKER_01")]
