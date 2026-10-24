from omni2.jobs.stt_worker import _groups
from omni2.stt.qwen3_asr_engine import attach_punctuation


def test_attach_punctuation_japanese():
    text = "皆さん、こんばんは。今日は「配信」です！"
    items = [("皆", 0.0, 0.1), ("さん", 0.1, 0.3), ("こんばんは", 0.4, 1.0), ("今日", 1.2, 1.4),
             ("は", 1.4, 1.5), ("配信", 1.6, 2.0), ("です", 2.0, 2.3)]
    words = attach_punctuation(text, items)
    assert "".join(w.text for w in words) == text
    assert [w.text for w in words] == ["皆", "さん、", "こんばんは。", "今日", "は", "「配信」", "です！"]
    assert words[2].start == 0.4


def test_attach_punctuation_english_keeps_spaces():
    text = "Hi everyone, and welcome back."
    items = [("Hi", 0, 1), ("everyone", 1, 2), ("and", 2, 3), ("welcome", 3, 4), ("back", 4, 5)]
    words = attach_punctuation(text, items)
    assert [w.text for w in words] == ["Hi", " everyone,", " and", " welcome", " back."]


def test_attach_punctuation_tolerates_mismatch():
    words = attach_punctuation("abc def", [("abc", 0, 1), ("xyz", 1, 2), ("def", 2, 3)])
    assert [w.text for w in words] == ["abc", "xyz", " def"]


def _item(job, seconds, lang="ja"):
    return {"job": job, "duration": seconds, "language": lang, "vad": "file", "hotwords": []}


def test_groups_batch_files_up_to_limit():
    items = [_item("a", 3000), _item("b", 3000), _item("c", 3000), _item("d", 100, "en")]
    groups = _groups(items, batches_files=True)
    assert [[i["job"] for i in g] for g in groups] == [["a", "b"], ["c"], ["d"]]


def test_groups_one_file_each_when_engine_cannot_batch():
    assert len(_groups([_item("a", 1), _item("b", 1)], batches_files=False)) == 2
