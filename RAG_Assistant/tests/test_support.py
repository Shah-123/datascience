import pytest

from rag_assistant.evaluation.calibration import FaultInjector, calibrate, sensitivity
from rag_assistant.evaluation.support import score_answer, split_claims, strip_citations

CTX = ["A late payment fee of $75 is charged when the balance is not paid in full by the due date."]


def test_verbatim_and_paraphrased_claims_are_supported():
    assert not score_answer("A late payment fee of $75 is charged. [1]", CTX).unfaithful
    assert not score_answer("The late payment fee is $75 when the balance is not paid by the due date.", CTX).unfaithful


def test_changed_number_is_unsupported():
    f = score_answer("A late payment fee of $85 is charged when the balance is not paid.", CTX)
    assert f.unfaithful and "85" in f.claims[0].reason


def test_invented_sentence_is_unsupported_and_scores_partially():
    f = score_answer("A late payment fee of $75 is charged. Students may appeal to the Dean of Students.", CTX)
    assert f.unfaithful and f.score == 0.5 and len(f.unsupported) == 1


def test_negation_flip_is_caught():
    ctx = ["International students are not eligible for Federal Work-Study."]
    assert score_answer("International students are eligible for Federal Work-Study.", ctx).unfaithful
    assert not score_answer("International students are not eligible for Federal Work-Study.", ctx).unfaithful


def test_determiner_no_is_not_treated_as_a_negation_flip():
    ctx = ["To register for more than 18 credits you need approval. No undergraduate may register for more than 21 credits."]
    assert not score_answer("You need approval to register for more than 18 credits, and undergraduates may register for at most 21 credits.", ctx).unfaithful


def test_claim_splitting_strips_markers_bullets_and_hedges():
    claims = split_claims("According to the policy, passwords expire every 180 days. [1]\n- Yes, fees apply per term.")
    assert claims[0].startswith("passwords expire") and claims[1].startswith("fees apply")
    assert strip_citations("a [1] b [2, 3]") == "a b"


def test_empty_answer_has_no_claims_and_is_faithful():
    f = score_answer("", CTX)
    assert f.claims == [] and f.score == 1.0 and not f.unfaithful


def test_calibration_set_has_no_unexpected_errors():
    c = calibrate()
    assert c.unexpected == []  # every remaining error is a documented, expected blind spot
    assert c.accuracy >= 0.75 and c.unfaithful_precision >= 0.8


def test_calibration_default_threshold_sits_on_a_plateau():
    accs = [calibrate(threshold=t).accuracy for t in (0.6, 0.7, 0.75, 0.8)]
    assert max(accs) - min(accs) < 0.05


def test_fault_injection_is_detected(sections, items):
    from rag_assistant import PipelineConfig, RAGPipeline
    pipe = RAGPipeline(sections, PipelineConfig(retriever="bm25", min_coverage=0.3))
    res, _ = sensitivity(pipe, items, rate=0.5)
    assert res.injected >= 15
    assert res.recall >= 0.85 and res.false_alarm_rate <= 0.1


def test_fault_injector_is_deterministic_and_leaves_abstentions_alone(pipeline):
    q = "What is the late payment fee?"
    hits = pipeline.retrieve(q)
    a = FaultInjector(pipeline.generator, rate=1.0).generate(q, hits)
    b = FaultInjector(pipeline.generator, rate=1.0).generate(q, hits)
    assert a.text == b.text and a.meta["fault"] in ("number", "invented", "negation") and a.text != pipeline.generator.generate(q, hits).text
    off = pipeline.retrieve("Who won the World Cup?")
    assert FaultInjector(pipeline.generator, rate=1.0).generate("Who won the World Cup?", off).abstained
