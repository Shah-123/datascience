import json

import pytest

from rag_assistant.cli import main


@pytest.fixture(autouse=True)
def no_llm_keys(monkeypatch):
    for k in ("MODELSCOPE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "RAG_LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RAG_ENV_FILE", "/nonexistent/.env")


def test_check_data_passes(capsys):
    assert main(["check-data"]) == 0
    assert "consistent" in capsys.readouterr().out


def test_ask_answers_with_sources(capsys):
    assert main(["ask", "--config", "default", "What is the late payment fee?"]) == 0
    out = capsys.readouterr().out
    assert "$75" in out and "sources:" in out


def test_ask_refuses_off_topic_questions(capsys):
    main(["ask", "--config", "default", "How do I bake sourdough bread?"])
    assert "REFUSED" in capsys.readouterr().out


def test_llm_generator_without_a_provider_exits_with_instructions():
    with pytest.raises(SystemExit) as exc:
        main(["ask", "--generator", "llm", "hello"])
    assert "MODELSCOPE_API_KEY" in str(exc.value)


def test_calibrate_command(capsys):
    assert main(["calibrate"]) == 0
    assert "accuracy" in capsys.readouterr().out


def test_eval_writes_outputs_and_gate_can_fail(tmp_path, capsys):
    out = tmp_path / "run"
    ok_gates = tmp_path / "ok.json"
    ok_gates.write_text(json.dumps({"hit@5": {"min": 0.5}}))
    assert main(["eval", "--config", "default", "--split", "test", "--out", str(out), "--gate", str(ok_gates)]) == 0
    assert (out / "summary.md").exists() and (out / "per_question.csv").exists() and (out / "metrics.json").exists()
    assert "quality gate passed" in capsys.readouterr().out

    bad_gates = tmp_path / "bad.json"
    bad_gates.write_text(json.dumps({"hit@5": {"min": 1.01}, "hallucination_rate": {"max": -0.1}}))
    assert main(["eval", "--config", "default", "--split", "test", "--out", str(out), "--gate", str(bad_gates)]) == 1
    assert "QUALITY GATE FAILED" in capsys.readouterr().out


def test_llm_models_needs_an_openai_compatible_provider():
    with pytest.raises(SystemExit):
        main(["llm-models"])


def test_missing_gate_file_is_a_clear_error_not_a_traceback(tmp_path, capsys):
    code = main(["eval", "--config", "default", "--split", "test", "--out", str(tmp_path / "o"), "--gate", str(tmp_path / "nope.json")])
    assert code == 2
    assert "cannot read the quality-gate file" in capsys.readouterr().err


def test_report_without_a_cached_study_explains_what_to_do(tmp_path, capsys):
    assert main(["report", "--out", str(tmp_path)]) == 2
    assert "run `python -m rag_assistant study` first" in capsys.readouterr().err
