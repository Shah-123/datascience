import math

import pytest

from rag_assistant.evaluation.metrics import (balanced_ci, bootstrap_ci, citation_scores, fact_recall,
                                              phrase_in, retrieval_metrics, token_f1)
from rag_assistant.ingest import Chunk
from rag_assistant.types import Answer, Hit


def hit(rank, text, cid=None):
    return Hit(Chunk(cid or f"c{rank}", "d", "D", "H", "d#h", text, 0), 1.0 / rank, rank)


def test_retrieval_metrics_hand_computed():
    hits = [hit(1, "nothing"), hit(2, "the late fee is $75"), hit(3, "x"), hit(4, "y"), hit(5, "z")]
    m = retrieval_metrics(hits, ["late fee is $75"], ks=(1, 3, 5), n_relevant_total=1)
    assert m["hit@1"] == 0 and m["hit@3"] == 1 and m["recall@3"] == 1
    assert m["mrr"] == pytest.approx(0.5)
    assert m["ndcg@5"] == pytest.approx(1 / math.log2(3))  # relevant at rank 2, ideal at rank 1


def test_recall_counts_partial_multihop_evidence():
    hits = [hit(1, "alpha fact"), hit(2, "unrelated")]
    m = retrieval_metrics(hits, ["alpha fact", "beta fact"], ks=(2,), n_relevant_total=2)
    assert m["recall@2"] == 0.5 and m["hit@2"] == 1


def test_no_relevant_chunk_gives_zero():
    m = retrieval_metrics([hit(1, "a"), hit(2, "b")], ["missing"], ks=(2,), n_relevant_total=1)
    assert m["mrr"] == 0 and m["ndcg@5"] == 0 and m["hit@2"] == 0


def test_phrase_matching_is_boundary_aware():
    assert phrase_in("fee of $75 applies", "fee of $75")
    assert not phrase_in("fee of $750", "fee of $75")


def test_fact_recall_ignores_citation_markers_and_accepts_alternatives():
    assert fact_recall("It costs $75 [1].", [["$75"]]) == 1
    assert fact_recall("Costs 75 dollars", [["$75", "75 dollars"]]) == 1
    assert fact_recall("see [1]", [["1"]]) == 0  # a bare citation marker must not satisfy a fact
    assert fact_recall("A and B", [["a"], ["z"]]) == 0.5


def test_token_f1():
    assert token_f1("late fee 75", "late fee 75") == pytest.approx(1)
    assert token_f1("completely different words", "late fee 75") == 0
    assert 0 < token_f1("late fee is 75 dollars total", "late fee 75") < 1


def test_citation_scores():
    ctx = [hit(1, "the fee is $75", "good"), hit(2, "irrelevant", "bad")]
    ans = Answer("q", "a", False, ["good", "bad"], ctx, 1.0)
    assert citation_scores(ans, ["fee is $75"]) == (0.5, 1.0)
    assert citation_scores(Answer("q", "a", False, [], ctx, 1.0), ["fee is $75"]) == (0.0, 0.0)


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    vals = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    lo, hi = bootstrap_ci(vals)
    assert (lo, hi) == bootstrap_ci(vals)
    assert lo <= sum(vals) / len(vals) <= hi and 0 <= lo and hi <= 1
    assert bootstrap_ci([1.0] * 5) == (1.0, 1.0)
    assert math.isnan(bootstrap_ci([])[0])


def test_balanced_ci():
    mean, lo, hi = balanced_ci([1, 1, 0, 0], [1, 1, 1, 1])
    assert mean == pytest.approx(0.75) and lo <= mean <= hi
