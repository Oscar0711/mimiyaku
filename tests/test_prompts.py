import json

from omni2.translate import prompts
from omni2.translate.client import ChatResult
from omni2.translate.glossary import Glossary, Term
from omni2.translate.translator import BatchTranslator


def test_style_detection():
    assert prompts.style_name_for(r"C:\m\HY-MT2-7B-Q6_K.gguf") == "generic"
    assert prompts.style_name_for(r"C:\m\HY-MT2-7B-Q6_K.gguf", "hy-mt") == "hy-mt"
    assert prompts.style_name_for("translategemma-12b-it.Q4_K_M.gguf") == "translategemma"
    assert prompts.style_name_for("Qwen3-8B-Q4_K_M.gguf") == "generic"
    assert prompts.style_name_for("Qwen3-8B-Q4_K_M.gguf", "hy-mt") == "hy-mt"


def test_hy_mt_request_uses_official_format():
    style = prompts.make_style("hy-mt", "ja", "zh-TW")
    req = style.build({"1": "こんにちは"}, context=[("前", "之前")], terms=[Term("東京", "東京")], summary="")
    text = req.messages[0]["content"]
    assert len(req.messages) == 1 and req.messages[0]["role"] == "user"
    assert "東京 翻译成 東京" in text
    assert "【背景信息】" in text and "前 → 之前" in text
    assert "翻译为繁体中文" in text
    assert req.schema["required"] == ["1"]
    assert req.sampling["top_k"] == 20


class SingleLineClient:
    def __init__(self):
        self.prompts = []

    def complete(self, req, *, temperature, max_tokens):
        assert req.messages is None and req.schema is None
        self.prompts.append(req.prompt)
        source = req.prompt.split("\n\n\n", 1)[1].split("<end_of_turn>", 1)[0]
        return ChatResult(f"譯:{source}\n", "stop")


def test_translategemma_translates_one_line_at_a_time():
    client = SingleLineClient()
    t = BatchTranslator(client, src_lang="en", tgt_lang="zh-TW", style="translategemma", opencc="")
    assert t.translate_all(["hello", "bye"]) == ["譯:hello", "譯:bye"]
    assert len(client.prompts) == 2
    assert "English (en) to Traditional Chinese (zh-TW)" in client.prompts[0]
    assert client.prompts[0].endswith("<start_of_turn>model\n")
