import pytest

from rag_assistant import REFUSAL_TEXT, PipelineConfig, RAGPipeline
from rag_assistant.evaluation import evaluate
from rag_assistant.evaluation.ablation import assign_folds, merge_runs, paired_delta
from rag_assistant.evaluation.gate import check_gates, make_gates
from rag_assistant.types import Answer


class Scripted:
    """A generator whose behaviour the test controls."""

    def __init__(self, mode):
        self.mode, self.name = mode, f"scripted-{mode}"

    def generate(self, question, hits):
        if self.mode == "abstain":
            return Answer(question, REFUSAL_TEXT, True, [], list(hits), 0.0)
        text = {"right": "A late payment fee of $75 is charged. [1]", "wrong": "The deposit is $999. [1]"}[self.mode]
        return Answer(question, text, False, [hits[0].chunk.chunk_id], list(hits), 1.0)


def run(sections, items, mode, retriever="bm25"):
    pipe = RAGPipeline(sections, PipelineConfig(retriever=retriever))
    pipe.generator = Scripted(mode)
    return evaluate(pipe, items)


def pick(items, *ids):
    return [i for i in items if i.id in ids]


def test_outcome_taxonomy_for_each_scenario(sections, items):
    late_fee = pick(items, "f04")  # answerable; evidence is easy to retrieve
    off = pick(items, "uo01")
    assert run(sections, late_fee, "right").results[0].outcome == "correct"
    assert run(sections, late_fee, "wrong").results[0].outcome == "wrong_answer_generation"
    assert run(sections, late_fee, "abstain").results[0].outcome == "false_refusal_evidence_present"
    assert run(sections, off, "abstain").results[0].outcome == "correct_abstention"
    assert run(sections, off, "right").results[0].outcome == "over_answer"
    # with a retriever that cannot find the evidence, the same failures are attributed to retrieval
    assert run(sections, late_fee, "wrong", "random").results[0].outcome == "wrong_answer_retrieval"
    assert run(sections, late_fee, "abstain", "random").results[0].outcome == "false_refusal_retrieval"


def test_metrics_reflect_scenarios(sections, items):
    subset = pick(items, "f04", "f05", "uo01", "uo02")
    s = run(sections, subset, "right").summary(n_boot=200)
    assert s["hit@5"]["mean"] == 1.0
    assert s["abstention_accuracy"]["mean"] == 0.0  # answered both unanswerable questions
    assert s["hallucination_rate"]["mean"] >= 0.5  # over-answers count as hallucination
    assert s["citation_precision"]["mean"] > 0


def test_always_abstaining_is_safe_but_useless(sections, items):
    s = run(sections, items, "abstain").summary(n_boot=200)
    assert s["abstention_accuracy"]["mean"] == 1.0 and s["answer_accuracy"]["mean"] == 0.0
    assert s["hallucination_rate"]["mean"] == 0.0
    assert s["balanced_score"]["mean"] == pytest.approx(0.5)  # abstention alone can never look good


def test_wrong_but_faithful_answer_is_not_called_a_hallucination_when_supported(sections, items):
    # "$999" appears nowhere in the context, so it IS unfaithful; a faithful-but-wrong answer would not be.
    r = run(sections, pick(items, "f04"), "wrong").results[0]
    assert r.faithfulness.unfaithful and r.metrics["hallucination_rate"] == 1.0


def test_default_pipeline_beats_random_on_every_retrieval_metric(sections, items):
    good = evaluate(RAGPipeline(sections, PipelineConfig(retriever="bm25")), items).summary(n_boot=0)
    rand = evaluate(RAGPipeline(sections, PipelineConfig(retriever="random")), items).summary(n_boot=0)
    for k in ("hit@5", "recall@5", "mrr", "ndcg@5"):
        assert good[k]["mean"] > rand[k]["mean"] + 0.5


def test_paired_delta_and_merge(sections, items):
    a, b = run(sections, items, "abstain"), run(sections, items, "right")
    d, lo, hi = paired_delta(a, b, "hallucination_rate")
    assert lo <= d <= hi and d > 0
    assert len(merge_runs([a, b], "m").results) == 2 * len(items)


def test_folds_are_stratified_disjoint_and_cover_everything(items):
    folds = assign_folds(items, 5)
    assert set(folds) == {i.id for i in items} and set(folds.values()) == set(range(5))
    for t in {i.type for i in items}:
        counts = [sum(1 for i in items if i.type == t and folds[i.id] == f) for f in range(5)]
        assert max(counts) - min(counts) <= 1


def test_quality_gates():
    summary = {"hit@5": {"mean": 0.9}, "hallucination_rate": {"mean": 0.1}}
    gates = make_gates(summary, margin=0.05)
    assert gates == {"hit@5": {"min": 0.85}, "hallucination_rate": {"max": 0.15}}
    assert check_gates(summary, gates) == []
    worse = {"hit@5": {"mean": 0.7}, "hallucination_rate": {"mean": 0.3}}
    failures = check_gates(worse, gates)
    assert len(failures) == 2 and "below the floor" in failures[0] and "above the ceiling" in failures[1]
    assert check_gates({}, gates) == ["hit@5: not measured", "hallucination_rate: not measured"]
