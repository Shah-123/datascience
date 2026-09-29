"""The OpenAI-compatible client (used for ModelScope) is exercised against a real local HTTP server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rag_assistant.config import load_env_file
from rag_assistant.llm import (DEFAULT_MODELSCOPE_MODEL, LLMError, MODELSCOPE_BASE_URL, OpenAICompatClient,
                               detect_provider, make_llm)

KEY = "ms-test-secret-key-123"


class State:
    fail_first = 0  # respond 500 this many times before succeeding
    seen: list = []
    mode = "ok"


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.headers.get("Authorization") != f"Bearer {KEY}":
                return self._json(401, {"error": "bad key"})
            self._json(200, {"data": [{"id": "b-model"}, {"id": "a-model"}]})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.seen.append((self.path, self.headers.get("Authorization"), payload))
            if self.headers.get("Authorization") != f"Bearer {KEY}":
                return self._json(401, {"error": "bad key"})
            if state.fail_first > 0:
                state.fail_first -= 1
                return self._json(500, {"error": "temporary"})
            if state.mode == "bad_request":
                return self._json(400, {"error": "unsupported parameter"})
            if not payload.get("stream"):
                return self._json(200, {"choices": [{"message": {"content": "  The fee is $75 [1].  "}}]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ({"reasoning_content": "thinking..."}, {"content": "The fee "}, {"content": "is $75 [1]."}):
                self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": piece}]}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    return Handler


@pytest.fixture
def server():
    state = State()
    state.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{httpd.server_port}/v1"
    yield state
    httpd.shutdown()


def client(server, **kw):
    kw.setdefault("sleep", lambda s: None)
    return OpenAICompatClient(server.url, kw.pop("key", KEY), "test-model", "modelscope", **kw)


def test_streaming_completion_ignores_reasoning_and_joins_content(server):
    out = client(server).complete("sys", "user")
    assert out == "The fee is $75 [1]."
    path, auth, payload = server.seen[0]
    assert path == "/v1/chat/completions" and payload["stream"] is True and payload["temperature"] == 0
    assert payload["messages"][0] == {"role": "system", "content": "sys"}


def test_non_streaming_completion(server):
    assert client(server, stream=False).complete("s", "u") == "The fee is $75 [1]."


def test_extra_body_is_forwarded(server):
    client(server, extra_body={"enable_thinking": False}).complete("s", "u")
    assert server.seen[0][2]["enable_thinking"] is False


def test_retries_on_server_errors_then_succeeds(server):
    server.fail_first = 2
    assert "$75" in client(server, max_retries=3).complete("s", "u")
    assert len(server.seen) == 3


def test_gives_up_after_max_retries(server):
    server.fail_first = 99
    with pytest.raises(LLMError, match="failed after 3 attempts"):
        client(server, max_retries=2).complete("s", "u")


def test_auth_failure_is_reported_without_leaking_the_key(server):
    with pytest.raises(LLMError) as exc:
        client(server, key="wrong-key-value").complete("s", "u")
    assert "authentication" in str(exc.value) and "wrong-key-value" not in str(exc.value)


def test_client_errors_are_not_retried(server):
    server.mode = "bad_request"
    with pytest.raises(LLMError, match="HTTP 400"):
        client(server, max_retries=3).complete("s", "u")
    assert len(server.seen) == 1


def test_connection_failure_is_an_llm_error():
    c = OpenAICompatClient("http://127.0.0.1:9/v1", KEY, "m", max_retries=1, timeout=1, sleep=lambda s: None)
    with pytest.raises(LLMError):
        c.complete("s", "u")


def test_list_models_sorted(server):
    assert client(server).list_models() == ["a-model", "b-model"]


def test_make_llm_selection_from_environment(monkeypatch):
    for k in ("MODELSCOPE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "RAG_LLM_PROVIDER", "RAG_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert make_llm() is None and detect_provider() is None
    monkeypatch.setenv("MODELSCOPE_API_KEY", KEY)
    assert detect_provider() == "modelscope"
    c = make_llm()
    assert c.base_url == MODELSCOPE_BASE_URL and c.model == DEFAULT_MODELSCOPE_MODEL
    monkeypatch.setenv("RAG_LLM_MODEL", "deepseek-ai/DeepSeek-V3.1")
    assert make_llm().model == "deepseek-ai/DeepSeek-V3.1"
    monkeypatch.setenv("RAG_LLM_EXTRA_JSON", '{"enable_thinking": false}')
    assert make_llm().extra_body == {"enable_thinking": False}
    monkeypatch.setenv("RAG_LLM_EXTRA_JSON", "not json")
    with pytest.raises(LLMError):
        make_llm()


def test_missing_key_and_unknown_provider_are_clear_errors(monkeypatch):
    monkeypatch.delenv("MODELSCOPE_API_KEY", raising=False)
    with pytest.raises(LLMError, match="MODELSCOPE_API_KEY"):
        make_llm("modelscope")
    with pytest.raises(LLMError, match="Unknown provider"):
        make_llm("bogus")


def test_env_file_loading_never_overrides_real_variables_or_blank_values(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nA_TEST_VAR=from-file\nB_TEST_VAR=\nexport C_TEST_VAR='quoted'\nREAL_VAR=file\n", encoding="utf-8")
    monkeypatch.setenv("REAL_VAR", "real")
    for k in ("A_TEST_VAR", "B_TEST_VAR", "C_TEST_VAR"):
        monkeypatch.delenv(k, raising=False)
    assert load_env_file(f)
    import os
    assert os.environ["A_TEST_VAR"] == "from-file" and os.environ["C_TEST_VAR"] == "quoted"
    assert "B_TEST_VAR" not in os.environ and os.environ["REAL_VAR"] == "real"
    monkeypatch.delenv("A_TEST_VAR"); monkeypatch.delenv("C_TEST_VAR")
    assert not load_env_file(tmp_path / "missing.env")


def test_anthropic_client_maps_errors_and_refusals(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    from rag_assistant.llm import AnthropicClient, LLMRefusal

    c = AnthropicClient(api_key="sk-test")

    class Block:
        type, text = "text", "hello"

    class Resp:
        stop_reason, content = "end_turn", [Block()]

    seen = {}

    def fake_create(**kw):
        seen.update(kw)
        return Resp()

    monkeypatch.setattr(c._client.messages, "create", fake_create)
    assert c.complete("sys", "user") == "hello"
    assert "temperature" not in seen and seen["max_tokens"] >= 4096 and seen["system"] == "sys"

    Resp.stop_reason = "refusal"
    with pytest.raises(LLMRefusal):
        c.complete("sys", "user")
