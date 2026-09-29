"""LLM providers behind one tiny interface: ``client.complete(system, user) -> str``.

Providers
  * ``anthropic``  - official Anthropic SDK.
  * ``modelscope`` - ModelScope API-Inference (OpenAI-compatible endpoint).
  * ``openai``     - any OpenAI-compatible server (OpenAI, vLLM, Ollama, ...).

Selection is by environment variable, so no credential ever appears in code:
  RAG_LLM_PROVIDER = anthropic | modelscope | openai   (auto-detected from available keys)
  RAG_LLM_MODEL    = model id for the chosen provider
  MODELSCOPE_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY
  RAG_LLM_EXTRA_JSON = extra JSON body fields, e.g. {"enable_thinking": false}
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable

MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
DEFAULT_MODELSCOPE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMError(RuntimeError):
    """The provider could not produce an answer (auth, network, quota, bad response)."""


class LLMRefusal(LLMError):
    """The model declined to answer (safety refusal)."""


class ChatClient:
    name = "base"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        raise NotImplementedError


class AnthropicClient(ChatClient):
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 max_retries: int = 3, timeout: float = 120.0):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMError("Provider 'anthropic' needs `pip install anthropic`.") from exc
        self._anthropic = anthropic
        self.model = model or os.environ.get("RAG_LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL
        self.name = f"anthropic:{self.model}"
        # api_key=None lets the SDK resolve credentials itself (env var or `ant auth login`).
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        a = self._anthropic
        try:
            # No temperature/top_p: the current Claude models reject sampling parameters.
            # max_tokens is generous because adaptive thinking tokens count towards it.
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max(max_tokens, 4096),
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except a.AuthenticationError as exc:
            raise LLMError("Anthropic authentication failed - check your API credentials.") from exc
        except a.RateLimitError as exc:
            raise LLMError("Anthropic rate limit reached; retry later.") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API (network problem).") from exc
        if resp.stop_reason == "refusal":
            raise LLMRefusal("The model declined to answer this request.")
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class OpenAICompatClient(ChatClient):
    """Minimal OpenAI-compatible chat client using only the standard library.

    Streams by default: several hosted models (notably hybrid-reasoning models on ModelScope)
    only accept streaming requests, and streaming also avoids long-request timeouts.
    """

    def __init__(self, base_url: str, api_key: str, model: str, provider: str = "openai",
                 timeout: float = 120.0, max_retries: int = 3, stream: bool = True,
                 extra_body: dict | None = None, sleep: Callable[[float], None] = time.sleep):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model, self.timeout, self.max_retries = model, timeout, max_retries
        self.stream, self.extra_body, self._sleep = stream, extra_body or {}, sleep
        self.name = f"{provider}:{model}"

    def _request(self, path: str, payload: dict | None = None):
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise LLMError(f"{self.name}: authentication/permission error (HTTP {exc.code}).") from exc
                if exc.code == 404:
                    raise LLMError(f"{self.name}: model or endpoint not found (HTTP 404).") from exc
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    body = exc.read().decode("utf-8", "replace")[:300]
                    raise LLMError(f"{self.name}: HTTP {exc.code}: {body}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
            if attempt < self.max_retries:
                self._sleep(min(2 ** attempt, 20))
        raise LLMError(f"{self.name}: request failed after {self.max_retries + 1} attempts: {last}")

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": self.stream,
            **self.extra_body,
        }
        with self._request("/chat/completions", payload) as resp:
            if not self.stream:
                body = json.loads(resp.read().decode("utf-8"))
                try:
                    return (body["choices"][0]["message"].get("content") or "").strip()
                except (KeyError, IndexError, TypeError) as exc:
                    raise LLMError(f"{self.name}: unexpected response shape: {str(body)[:200]}") from exc
            parts: list[str] = []
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (ValueError, KeyError, IndexError):
                    continue
                if delta.get("content"):  # reasoning_content (hidden chain-of-thought) is ignored
                    parts.append(delta["content"])
            return "".join(parts).strip()

    def list_models(self) -> list[str]:
        with self._request("/models") as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return sorted(m["id"] for m in body.get("data", []))


def _extra_body() -> dict:
    raw = os.environ.get("RAG_LLM_EXTRA_JSON")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise LLMError("RAG_LLM_EXTRA_JSON is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise LLMError("RAG_LLM_EXTRA_JSON must be a JSON object.")
    return value


def detect_provider() -> str | None:
    if os.environ.get("MODELSCOPE_API_KEY"):
        return "modelscope"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def make_llm(provider: str | None = None, model: str | None = None) -> ChatClient | None:
    """Build a client from the environment; returns None when no provider is configured (offline mode)."""
    provider = (provider or os.environ.get("RAG_LLM_PROVIDER") or detect_provider() or "").lower()
    if provider in ("", "none", "offline"):
        return None
    if provider == "anthropic":
        return AnthropicClient(model=model)
    model = model or os.environ.get("RAG_LLM_MODEL")
    if provider == "modelscope":
        key = os.environ.get("MODELSCOPE_API_KEY")
        if not key:
            raise LLMError("Set MODELSCOPE_API_KEY to use the modelscope provider.")
        return OpenAICompatClient(os.environ.get("MODELSCOPE_BASE_URL", MODELSCOPE_BASE_URL), key,
                                  model or DEFAULT_MODELSCOPE_MODEL, "modelscope", extra_body=_extra_body())
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("Set OPENAI_API_KEY to use the openai provider.")
        return OpenAICompatClient(os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), key,
                                  model or DEFAULT_OPENAI_MODEL, "openai", extra_body=_extra_body())
    raise LLMError(f"Unknown provider {provider!r}; use anthropic, modelscope or openai.")
