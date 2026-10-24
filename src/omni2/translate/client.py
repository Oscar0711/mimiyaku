"""OpenAI 相容的 chat completions client。llama-server、Ollama、LM Studio 都能用。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import httpx

from omni2.translate.prompts import Request


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None


class ChatClient:
    def __init__(self, base_url: str, *, model: str = "local", api_key: str | None = None,
                 timeout: float = 600.0):
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self._http = httpx.Client(base_url=base, timeout=timeout, headers=headers)

    def _body(self, messages: list[dict], *, temperature: float, max_tokens: int,
              json_schema: dict | None, stream: bool) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            # 以下兩個是 llama-server 的擴充欄位，其他服務會忽略
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": True, "schema": json_schema},
            }
        return body

    def complete(self, req: Request, *, temperature: float = 0.2, max_tokens: int = 1024) -> ChatResult:
        if req.prompt is not None:
            # 原始 prompt（不套聊天模板）走 completions
            body = {"model": self.model, "prompt": req.prompt, "temperature": temperature,
                    "max_tokens": max_tokens, "stream": False, "cache_prompt": True, **req.sampling}
            r = self._http.post("/v1/completions", json=body)
            r.raise_for_status()
            choice = r.json()["choices"][0]
            return ChatResult(choice.get("text") or "", choice.get("finish_reason"))
        body = self._body(req.messages, temperature=temperature, max_tokens=max_tokens,
                          json_schema=req.schema, stream=False)
        body.update(req.sampling)
        r = self._http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        return ChatResult(choice["message"].get("content") or "", choice.get("finish_reason"))

    def stream(self, messages: list[dict], *, temperature: float = 0.2,
               max_tokens: int = 1024) -> Iterator[str]:
        body = self._body(messages, temperature=temperature, max_tokens=max_tokens,
                          json_schema=None, stream=True)
        with self._http.stream("POST", "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    def close(self) -> None:
        self._http.close()
