"""Headless UI tests using Streamlit's AppTest (no browser needed)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def no_llm_keys(monkeypatch):
    for k in ("MODELSCOPE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RAG_ENV_FILE", "/nonexistent/.env")


def ask(question: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.text_input(key="question").set_value(question).run()
    return at


def test_app_loads_with_three_tabs():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Ask", "How well does it work?", "Corpus"]


def test_answerable_question_shows_answer_sources_and_faithfulness_check():
    at = ask("What is the late payment fee?")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "$75" in text
    assert any("Faithfulness check passed" in s.value for s in at.success)
    assert any("Sources cited" in h.value for h in at.subheader)


def test_unanswerable_question_is_refused_not_guessed():
    at = ask("Who won the 2022 FIFA World Cup?")
    assert not at.exception
    assert any("couldn't find this" in w.value for w in at.warning)
    assert not any("Faithfulness check" in s.value for s in at.success)


def test_choosing_llm_without_a_key_explains_and_falls_back():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio[0].set_value("LLM").run()
    assert not at.exception
    assert any("No LLM provider is configured" in e.value for e in at.error)
