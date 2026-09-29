"""Retrievers. All share one interface: ``fit(chunks)`` then ``search(query, k) -> [(index, score)]``.

Implemented without any model download so the whole system runs offline:
  * BM25      - lexical, exact-term matching (own implementation).
  * TF-IDF    - unigram+bigram cosine similarity.
  * LSA       - TF-IDF followed by truncated SVD: a cheap "semantic" retriever.
  * Hybrid    - reciprocal-rank fusion of any of the above.
  * Random    - seeded floor, so every metric has a "no skill" reference point.
  * Dense     - plug in any embedding function (e.g. sentence-transformers).
"""
from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Callable, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .ingest import Chunk
from .text import analyze, content_terms

Ranking = list[tuple[int, float]]


class Retriever:
    name = "base"
    add_heading = True

    def fit(self, chunks: Sequence[Chunk]) -> "Retriever":
        raise NotImplementedError

    def search(self, query: str, k: int) -> Ranking:
        raise NotImplementedError

    def _texts(self, chunks: Sequence[Chunk]) -> list[str]:
        return [c.index_text(self.add_heading) for c in chunks]


def _top_k(scores: np.ndarray, k: int) -> Ranking:
    if k >= len(scores):
        order = np.argsort(-scores, kind="stable")
    else:
        part = np.argpartition(-scores, k)[:k]
        order = part[np.argsort(-scores[part], kind="stable")]
    return [(int(i), float(scores[i])) for i in order[:k]]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, add_heading: bool = True):
        self.k1, self.b, self.add_heading = k1, b, add_heading

    def fit(self, chunks: Sequence[Chunk]) -> "BM25Retriever":
        docs = [content_terms(t) for t in self._texts(chunks)]
        self.n = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avg_len = float(self.lengths.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }
        return self

    def search(self, query: str, k: int) -> Ranking:
        scores = np.zeros(self.n)
        for term in set(content_terms(query)):
            for i, tf in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return _top_k(scores, k)


class TfidfRetriever(Retriever):
    name = "tfidf"

    def __init__(self, add_heading: bool = True):
        self.add_heading = add_heading

    def fit(self, chunks: Sequence[Chunk]) -> "TfidfRetriever":
        self.vec = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self._texts(chunks))
        return self

    def search(self, query: str, k: int) -> Ranking:
        q = self.vec.transform([query])
        return _top_k((self.matrix @ q.T).toarray().ravel(), k)


class LSARetriever(Retriever):
    """Latent semantic analysis: captures term co-occurrence, so it can match related words."""

    name = "lsa"

    def __init__(self, n_components: int = 96, add_heading: bool = True):
        self.n_components, self.add_heading = n_components, add_heading

    def fit(self, chunks: Sequence[Chunk]) -> "LSARetriever":
        self.vec = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
        tfidf = self.vec.fit_transform(self._texts(chunks))
        n_comp = max(2, min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
        self.matrix = normalize(self.svd.fit_transform(tfidf))
        return self

    def search(self, query: str, k: int) -> Ranking:
        q = normalize(self.svd.transform(self.vec.transform([query])))
        return _top_k(self.matrix @ q.ravel(), k)


class DenseRetriever(Retriever):
    """Bring-your-own embeddings: ``embed_fn(list[str]) -> (n, d) array`` (L2-normalised or not)."""

    def __init__(self, embed_fn: Callable[[list[str]], np.ndarray], name: str = "dense", add_heading: bool = True):
        self.embed_fn, self.name, self.add_heading = embed_fn, name, add_heading

    def fit(self, chunks: Sequence[Chunk]) -> "DenseRetriever":
        self.matrix = normalize(np.asarray(self.embed_fn(self._texts(chunks)), dtype=float))
        return self

    def search(self, query: str, k: int) -> Ranking:
        q = normalize(np.asarray(self.embed_fn([query]), dtype=float))[0]
        return _top_k(self.matrix @ q, k)


def sentence_transformer_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Thin wrapper for real embeddings. Requires ``pip install sentence-transformers`` + model download."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Dense retrieval needs `pip install sentence-transformers`.") from exc
    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class RandomRetriever(Retriever):
    """Floor baseline: a deterministic pseudo-random ranking per query."""

    name = "random"

    def fit(self, chunks: Sequence[Chunk]) -> "RandomRetriever":
        self.n = len(chunks)
        return self

    def search(self, query: str, k: int) -> Ranking:
        rng = np.random.default_rng(zlib.crc32(query.encode()))
        return _top_k(rng.random(self.n), k)


class HybridRetriever(Retriever):
    """Reciprocal-rank fusion: score(d) = sum_i w_i / (rrf_k + rank_i(d))."""

    def __init__(self, parts: Sequence[Retriever], weights: Sequence[float] | None = None,
                 rrf_k: int = 60, depth: int = 50):
        self.parts, self.rrf_k, self.depth = list(parts), rrf_k, depth
        self.weights = list(weights) if weights else [1.0] * len(self.parts)
        self.name = "hybrid(" + "+".join(p.name for p in self.parts) + ")"

    def fit(self, chunks: Sequence[Chunk]) -> "HybridRetriever":
        for p in self.parts:
            p.fit(chunks)
        return self

    def search(self, query: str, k: int) -> Ranking:
        fused: dict[int, float] = {}
        for w, part in zip(self.weights, self.parts):
            for rank, (idx, _) in enumerate(part.search(query, self.depth), start=1):
                fused[idx] = fused.get(idx, 0.0) + w / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda kv: -kv[1])[:k]


RETRIEVER_NAMES = ("bm25", "tfidf", "lsa", "glove", "hybrid", "random")


def build_retriever(spec: str = "hybrid", add_heading: bool = True) -> Retriever:
    """``bm25`` | ``tfidf`` | ``lsa`` | ``glove`` | ``random`` | ``hybrid`` (= bm25+lsa) | ``hybrid:bm25+glove``."""
    spec = spec.strip().lower()
    if spec.startswith("hybrid"):
        parts = spec.split(":", 1)[1] if ":" in spec else "bm25+lsa"
        return HybridRetriever([build_retriever(p, add_heading) for p in parts.split("+")])
    table = {"bm25": BM25Retriever, "tfidf": TfidfRetriever, "lsa": LSARetriever}
    if spec == "random":
        return RandomRetriever()
    if spec == "glove":
        from .embeddings import GloVeRetriever  # imported lazily: needs the downloaded vectors
        return GloVeRetriever(add_heading=add_heading)
    if spec in table:
        return table[spec](add_heading=add_heading)
    raise ValueError(f"Unknown retriever {spec!r}; choose from {RETRIEVER_NAMES}")


def retriever_choices(tuned: str, glove_ok: bool) -> list[str]:
    """Retrievers a UI can offer. The tuned one (what the evaluation measured) comes first when it is usable."""
    opts = ["bm25", "tfidf", "hybrid:bm25+lsa"]
    if glove_ok:
        opts += ["glove", "hybrid:bm25+glove", "hybrid:bm25+lsa+glove"]
    if tuned in opts:
        opts.remove(tuned)
        opts.insert(0, tuned)
    return opts
