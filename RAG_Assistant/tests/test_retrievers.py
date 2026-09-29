import numpy as np
import pytest

from rag_assistant.ingest import ChunkConfig, chunk_sections
from rag_assistant.retrievers import (BM25Retriever, DenseRetriever, HybridRetriever, LSARetriever, RandomRetriever,
                                      TfidfRetriever, build_retriever)


@pytest.fixture(scope="module")
def chunks(sections):
    return chunk_sections(sections, ChunkConfig())


@pytest.mark.parametrize("name", ["bm25", "tfidf", "lsa", "hybrid", "hybrid:bm25+tfidf"])
def test_every_retriever_finds_an_easy_answer(chunks, name):
    r = build_retriever(name).fit(chunks)
    top = [chunks[i] for i, _ in r.search("What is the late payment fee?", 3)]
    assert any("late payment fee of $75" in c.text for c in top)


def test_results_are_sorted_and_respect_k(chunks):
    r = BM25Retriever().fit(chunks)
    res = r.search("graduate admission deadline", 7)
    assert len(res) == 7
    assert [s for _, s in res] == sorted((s for _, s in res), reverse=True)


def test_unknown_query_terms_do_not_crash(chunks):
    for r in (BM25Retriever(), TfidfRetriever(), LSARetriever()):
        assert len(r.fit(chunks).search("zzzz qqqq", 3)) == 3


def test_random_retriever_is_deterministic_per_query(chunks):
    r = RandomRetriever().fit(chunks)
    assert r.search("abc", 5) == r.search("abc", 5)
    assert r.search("abc", 5) != r.search("xyz", 5)


def test_hybrid_rrf_math():
    class Fixed:
        def __init__(self, order, name):
            self.order, self.name = order, name

        def fit(self, chunks):
            return self

        def search(self, q, k):
            return [(i, 1.0) for i in self.order][:k]

    h = HybridRetriever([Fixed([0, 1, 2], "a"), Fixed([2, 1, 0], "b")], rrf_k=60)
    scores = dict(h.search("q", 3))
    assert scores[1] == pytest.approx(2 / 62)  # rank 2 in both lists
    assert scores[0] == pytest.approx(1 / 61 + 1 / 63)
    assert h.search("q", 1)[0][0] in (0, 2)


def test_dense_retriever_with_stub_embeddings(chunks):
    vocab = ["fee", "deadline", "housing", "library"]

    def embed(texts):
        return np.array([[t.lower().count(w) + 1e-3 for w in vocab] for t in texts], dtype=float)

    r = DenseRetriever(embed).fit(chunks)
    top = [chunks[i] for i, _ in r.search("library library library", 3)]
    assert any("library" in c.text.lower() for c in top)


def test_build_retriever_rejects_unknown_names():
    with pytest.raises(ValueError):
        build_retriever("nonsense")
