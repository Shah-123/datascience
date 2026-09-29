from rag_assistant.evaluation.judge import LLMJudge, build_judge_prompt, parse_judgement
from rag_assistant.llm import ChatClient, LLMError


class Fake(ChatClient):
    name = "fake"

    def __init__(self, reply=None, exc=None):
        self.reply, self.exc = reply, exc

    def complete(self, system, user, max_tokens=1024):
        if self.exc:
            raise self.exc
        return self.reply


def test_parse_accepts_clean_and_wrapped_json():
    assert parse_judgement('{"correct": true, "faithful": false, "unsupported_claims": ["x"]}')["unsupported_claims"] == ["x"]
    wrapped = 'Sure!\n```json\n{"correct": false, "faithful": true}\n```'
    assert parse_judgement(wrapped) == {"correct": False, "faithful": True, "unsupported_claims": []}


def test_parse_rejects_junk_instead_of_guessing():
    for junk in ("", "no json here", '{"correct": "yes", "faithful": true}', '{"correct": true}', "{broken"):
        assert parse_judgement(junk) is None


def test_judge_excludes_failed_calls_rather_than_guessing():
    assert LLMJudge(Fake(exc=LLMError("down")))("q", ["p"], "ref", "cand") is None
    assert LLMJudge(Fake("garbage"))("q", ["p"], "ref", "cand") is None
    ok = LLMJudge(Fake('{"correct": true, "faithful": true}'))("q", ["p"], "ref", "cand")
    assert ok["correct"] and ok["faithful"]


def test_prompt_marks_unanswerable_questions():
    p = build_judge_prompt("q?", ["passage one"], "", "cand")
    assert "[1] passage one" in p and "unanswerable" in p
