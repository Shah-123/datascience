"""Metric definitions. Every metric here is computed per question, then averaged (with bootstrap CIs).

Retrieval (answerable questions). "Evidence" = verbatim phrases the retriever must surface.
  hit@k      1 if any evidence phrase appears in the top-k chunks
  recall@k   fraction of evidence phrases appearing in the top-k chunks
  mrr        1 / rank of the first chunk that contains evidence
  ndcg@5     binary-relevance nDCG over the top 5 chunks
Because matching is on text, not chunk ids, chunking strategies are compared fairly.

Answer quality
  answer_accuracy      answered AND every required fact present
  false_refusal_rate   abstained on an answerable question
  abstention_accuracy  abstained on an unanswerable question
  unfaithful_rate      answers containing >=1 claim unsupported by the retrieved context
  hallucination_rate   over ALL questions: answered AND (question unanswerable OR unfaithful)
  citation_precision   cited chunks that actually contain evidence
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from typing import Sequence

import numpy as np

from ..text import content_terms, normalize_for_match
from ..types import Answer, Hit
from .support import strip_citations

_norm = lru_cache(maxsize=None)(normalize_for_match)


def phrase_in(text: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(text)


# ---------------------------------------------------------------------------- retrieval


def retrieval_metrics(hits: Sequence[Hit], evidence: Sequence[str], ks: Sequence[int],
                      n_relevant_total: int) -> dict[str, float]:
    rel = [any(phrase_in(h.chunk.text, p) for p in evidence) for h in hits]
    out: dict[str, float] = {}
    for k in ks:
        top = hits[:k]
        found = [p for p in evidence if any(phrase_in(h.chunk.text, p) for h in top)]
        out[f"hit@{k}"] = float(bool(found))
        out[f"recall@{k}"] = len(found) / len(evidence)
    first = next((i for i, r in enumerate(rel, start=1) if r), None)
    out["mrr"] = 1.0 / first if first else 0.0
    k = 5
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(rel[:k], start=1) if r)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(k, max(n_relevant_total, 1)) + 1))
    out["ndcg@5"] = dcg / idcg if idcg else 0.0
    return out


# ---------------------------------------------------------------------------- answers


def fact_recall(answer_text: str, facts: Sequence[Sequence[str]]) -> float:
    text = strip_citations(answer_text)
    return sum(any(phrase_in(text, alt) for alt in fact) for fact in facts) / len(facts) if facts else 1.0


def token_f1(prediction: str, reference: str) -> float:
    p, r = Counter(content_terms(strip_citations(prediction))), Counter(content_terms(reference))
    overlap = sum((p & r).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def citation_scores(answer: Answer, evidence: Sequence[str]) -> tuple[float, float]:
    """(precision, recall) of the chunks the answer cites, judged against the gold evidence."""
    by_id = {h.chunk.chunk_id: h.chunk for h in answer.contexts}
    cited = [by_id[c] for c in answer.citations if c in by_id]
    if not cited:
        return 0.0, 0.0
    precision = sum(any(phrase_in(c.text, p) for p in evidence) for c in cited) / len(cited)
    recall = sum(any(phrase_in(c.text, p) for c in cited) for p in evidence) / len(evidence)
    return precision, recall


def evidence_in_context(answer: Answer, evidence: Sequence[str]) -> bool:
    """Were ALL evidence phrases visible to the generator? (Separates retrieval from generation errors.)"""
    return all(any(phrase_in(h.chunk.text, p) for h in answer.contexts) for p in evidence)


# ---------------------------------------------------------------------------- statistics


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(arr), (n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def balanced_ci(answerable: Sequence[float], unanswerable: Sequence[float], n_boot: int = 2000,
                seed: int = 0) -> tuple[float, float, float]:
    """Macro-average of answer accuracy (answerable) and abstention accuracy (unanswerable)."""
    a, u = np.asarray(answerable, float), np.asarray(unanswerable, float)
    rng = np.random.default_rng(seed)
    ia, iu = rng.integers(0, len(a), (n_boot, len(a))), rng.integers(0, len(u), (n_boot, len(u)))
    boot = 0.5 * (a[ia].mean(axis=1) + u[iu].mean(axis=1))
    return float(0.5 * (a.mean() + u.mean())), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
