import pytest

from rag_assistant import REFUSAL_TEXT
from rag_assistant.generators import (LLMGenerator, build_prompt, expected_answer_type, satisfies_type)
from rag_assistant.llm import ChatClient, LLMError, LLMRefusal


def test_extractive_answers_and_cites(pipeline):
    ans = pipeline.ask("What is the late payment fee?")
    assert not ans.abstained and "$75" in ans.text
    assert ans.citations and ans.citations[0].startswith("tuition_fees_and_financial_aid#billing")


@pytest.mark.parametrize("q", ["Who won the 2022 FIFA World Cup?", "How do I bake sourdough bread?", "Explain photosynthesis."])
def test_extractive_abstains_on_off_topic(pipeline, q):
    ans = pipeline.ask(q)
    assert ans.abstained and ans.text == REFUSAL_TEXT and ans.citations == []


def test_answer_type_detection():
    assert expected_answer_type("How much does a transcript cost?") == "money"
    assert expected_answer_type("How many credits do I need?") == "count"
    assert expected_answer_type("When is the deadline?") == "date"
    assert expected_answer_type("Until what time is the library open?") == "time"
    assert expected_answer_type("Can I keep a cat?") is None


def test_answer_type_satisfaction():
    assert satisfies_type("A late fee of $75 is charged.", "money")
    assert not satisfies_type("Room rates are charged per term.", "money")
    assert satisfies_type("Applications are due November 1.", "date")
    assert not satisfies_type("Awarded to first-year students with a GPA of 3.8.", "date")
    assert satisfies_type("It runs until 2 a.m.", "time")


def test_type_gate_can_be_switched_off(sections):
    from rag_assistant import PipelineConfig, RAGPipeline
    q = "What is the fee for ordering a duplicate diploma?"
    gated = RAGPipeline(sections, PipelineConfig(retriever="bm25", min_coverage=0.0, type_gate=True)).ask(q)
    ungated = RAGPipeline(sections, PipelineConfig(retriever="bm25", min_coverage=0.0, type_gate=False)).ask(q)
    assert not ungated.abstained  # without the gate it quotes something topical
    assert gated.abstained or "$" in gated.text  # with the gate it only answers when an amount is present


class FakeClient(ChatClient):
    name = "fake"

    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def complete(self, system, user, max_tokens=1024):
        self.calls.append((system, user))
        if self.exc:
            raise self.exc
        return self.reply


def _hits(pipeline, q):
    return pipeline.retrieve(q, 3)


def test_llm_generator_parses_citations(pipeline):
    hits = _hits(pipeline, "What is the late payment fee?")
    gen = LLMGenerator(FakeClient("The late fee is $75 [1][3]."))
    ans = gen.generate("q", hits)
    assert not ans.abstained
    assert ans.citations == [hits[0].chunk.chunk_id, hits[2].chunk.chunk_id]


def test_llm_generator_ignores_out_of_range_citations(pipeline):
    ans = LLMGenerator(FakeClient("Yes [9].")).generate("q", _hits(pipeline, "late fee"))
    assert ans.citations == []


@pytest.mark.parametrize("reply", ["NOT_FOUND", "not_found - nothing about that", ""])
def test_llm_generator_abstains_on_not_found(pipeline, reply):
    ans = LLMGenerator(FakeClient(reply)).generate("q", _hits(pipeline, "late fee"))
    assert ans.abstained and ans.text == REFUSAL_TEXT


def test_llm_generator_fails_closed_on_errors_and_refusals(pipeline):
    err = LLMGenerator(FakeClient(exc=LLMError("boom"))).generate("q", _hits(pipeline, "late fee"))
    assert err.abstained and err.meta["error"] == "boom"
    ref = LLMGenerator(FakeClient(exc=LLMRefusal("no"))).generate("q", _hits(pipeline, "late fee"))
    assert ref.abstained and ref.meta["llm_refusal"]


def test_prompt_numbers_passages_and_includes_question(pipeline):
    hits = _hits(pipeline, "late fee")
    prompt = build_prompt("What is the fee?", hits)
    assert "[1] (" in prompt and "[3] (" in prompt and prompt.endswith("Question: What is the fee?")


def test_llm_pipeline_requires_a_client(sections):
    from rag_assistant import PipelineConfig, RAGPipeline
    with pytest.raises(ValueError):
        RAGPipeline(sections, PipelineConfig(generator="llm"))
