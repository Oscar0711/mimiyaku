import json

import numpy as np

from omni2.post.diarize import assign_short, cluster
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


def test_assign_short_uses_nearest_voice_then_neighbours():
    a, b = _voice(1, 3), _voice(2, 3)
    labels = [0, None, 1, None, None]
    embeddings = [a[0], b[1], b[0], None, a[1]]
    assert assign_short(labels, embeddings) == [0, 1, 1, 1, 0]
    assert assign_short([None, None], [None, None]) == [0, 0]


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
