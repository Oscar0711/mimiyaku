import json
from collections import Counter

from omni2.translate.client import ChatResult
from omni2.translate.glossary import Glossary, Term
from omni2.translate.translator import BatchTranslator, make_converter, tidy


class FakeClient:
    """模擬模型：把每句翻成「譯:原文」，並依原文裡的關鍵字模擬各種錯誤。"""

    def __init__(self, truncate_over: int | None = None):
        self.calls: list[dict[str, str]] = []
        self.prompts: list[str] = []
        self.seen = Counter()
        self.truncate_over = truncate_over

    def complete(self, req, *, temperature, max_tokens):
        user = req.messages[-1]["content"]
        self.prompts.append(user)
        lines = json.loads(user.split("【待翻譯】\n", 1)[1])
        self.calls.append(lines)
        if self.truncate_over and len(lines) > self.truncate_over:
            return ChatResult('{"1": "半', "length")
        out = {}
        for k, src in lines.items():
            self.seen[src] += 1
            if "KANA" in src and self.seen[src] == 1:
                out[k] = "これはテスト"
            elif "ALWAYS_BAD" in src:
                out[k] = "ずっとダメ"
            elif "SIMPLIFIED" in src:
                out[k] = "这个软件"
            else:
                out[k] = f"譯:{src}"
        return ChatResult(json.dumps(out, ensure_ascii=False), "stop")


def make(client, **kw):
    kw.setdefault("opencc", "")
    return BatchTranslator(client, src_lang="ja", tgt_lang="zh-TW", **kw)


def test_aligned_batches_and_checkpoints():
    client = FakeClient()
    texts = [f"line{i}" for i in range(25)]
    saved = []
    out = make(client).translate_all(texts, on_result=lambda d: saved.append(dict(d)))
    assert out == [f"譯:line{i}" for i in range(25)]
    assert [len(c) for c in client.calls] == [10, 10, 5]
    assert [len(s) for s in saved] == [10, 10, 5]


def test_resume_only_translates_missing():
    client = FakeClient()
    texts = [f"line{i}" for i in range(12)]
    done = {i: f"舊{i}" for i in range(10)}
    out = make(client).translate_all(texts, done=done)
    assert out[:10] == [f"舊{i}" for i in range(10)]
    assert out[10:] == ["譯:line10", "譯:line11"]
    assert client.calls == [{"1": "line10", "2": "line11"}]


def test_context_uses_previous_pairs():
    client = FakeClient()
    make(client, batch_lines=2, context_lines=1).translate_all(["a", "b", "c"])
    assert "原：b\n譯：譯:b" in client.prompts[1]


def test_duplicates_are_translated_once():
    client = FakeClient()
    out = make(client).translate_all(["yes", "yes", "no", "Yes"])
    assert out == ["譯:yes", "譯:yes", "譯:no", "譯:yes"]
    assert client.seen["yes"] == 1


def test_bad_line_is_retried_alone():
    client = FakeClient()
    t = make(client)
    out = t.translate_all(["ok", "KANA line", "ok2"])
    assert out == ["譯:ok", "譯:KANA line", "譯:ok2"]
    assert client.calls[-1] == {"1": "KANA line"}
    assert t.issues == []


def test_persistent_failure_is_reported():
    t = make(FakeClient(), max_retries=2)
    out = t.translate_all(["ALWAYS_BAD"])
    assert out == ["ずっとダメ"]
    assert [i.reason for i in t.issues] == ["夾雜日文假名"]


def test_truncated_batch_is_split():
    client = FakeClient(truncate_over=3)
    out = make(client).translate_all([f"s{i}" for i in range(8)])
    assert out == [f"譯:s{i}" for i in range(8)]
    assert sorted(len(c) for c in client.calls if len(c) <= 3) == [2, 2, 2, 2]


def test_same_language_skips_model():
    client = FakeClient()
    t = BatchTranslator(client, src_lang="zh-CN", tgt_lang="zh-TW", opencc="s2twp")
    assert t.translate_all(["这个软件"]) == ["這個軟體"]
    assert client.calls == []


def test_opencc_fixes_simplified_output():
    t = make(FakeClient(), opencc="s2twp")
    assert t.translate_all(["SIMPLIFIED"]) == ["這個軟體"]


def test_converter_keeps_correct_traditional():
    convert = make_converter("s2twp", "zh-TW")
    assert convert("湯頭超濃郁的") == "湯頭超濃郁的"
    assert convert("皇后住在一公里外") == "皇后住在一公里外"
    assert convert("这个软件的内存") == "這個軟體的記憶體"
    assert convert("這個网站很好用") == "這個網站很好用"
    assert convert("不過完全沒進步，有點气馁") == "不過完全沒進步，有點氣餒"
    assert convert("Hello 123") == "Hello 123"
    assert make_converter("s2twp", "en")("这个") == "这个"


def test_tidy_drops_unbalanced_quotes():
    assert tidy("下次我們在下週三再確認吧。」", "次回は確認しましょう。") == "下次我們在下週三再確認吧。"
    assert tidy("「你好", "こんにちは") == "你好"
    assert tidy("他說「好」。", "彼は「いい」と言った。") == "他說「好」。"
    # 原文本身就不成對時（字幕跨行）照原樣保留
    assert tidy("他說「明天", "彼は「明日") == "他說「明天"


def test_glossary_terms_in_prompt():
    client = FakeClient()
    g = Glossary([Term("ホロライブ", "hololive", "事務所"), Term("無関係", "無關")])
    make(client, glossary=g).translate_all(["ホロライブの配信"])
    assert "ホロライブ → hololive（事務所）" in client.prompts[0]
    assert "無関係" not in client.prompts[0]
