"""Run a pipeline over the golden set and turn raw answers into metrics + a failure taxonomy."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..pipeline import RAGPipeline
from ..types import Answer, Hit
from .dataset import EvalItem
from .metrics import (balanced_ci, bootstrap_ci, citation_scores, evidence_in_context, fact_recall,
                      phrase_in, retrieval_metrics, token_f1)
from .support import DEFAULT_THRESHOLD, Faithfulness, score_answer

DEFAULT_KS = (1, 3, 5, 8)

# Outcome taxonomy - every question lands in exactly one bucket.
OUTCOMES_ANSWERABLE = ("correct", "wrong_answer_generation", "wrong_answer_retrieval",
                       "false_refusal_evidence_present", "false_refusal_retrieval")
OUTCOMES_UNANSWERABLE = ("correct_abstention", "over_answer")
OUTCOME_LABELS = {
    "correct": "Correct",
    "wrong_answer_generation": "Wrong answer - evidence was retrieved",
    "wrong_answer_retrieval": "Wrong answer - retrieval missed evidence",
    "false_refusal_evidence_present": "Refused - evidence was retrieved",
    "false_refusal_retrieval": "Refused - retrieval missed evidence",
    "correct_abstention": "Correctly refused",
    "over_answer": "Answered an unanswerable question",
}


@dataclass
class ItemResult:
    item: EvalItem
    answer: Answer
    metrics: dict[str, float]  # only the metrics that apply to this item
    outcome: str
    faithfulness: Faithfulness | None = None
    judge: dict | None = None

    def to_row(self) -> dict:
        a = self.answer
        return {
            "id": self.item.id, "type": self.item.type, "split": self.item.split,
            "question": self.item.question, "outcome": self.outcome, "abstained": a.abstained,
            "answer": a.text, "reference": self.item.answer,
            "top_chunks": ";".join(h.chunk.chunk_id for h in a.contexts[:3]),
            "unsupported_claims": " | ".join(self.faithfulness.unsupported) if self.faithfulness else "",
            **{k: round(v, 4) for k, v in self.metrics.items()},
        }


@dataclass
class EvalRun:
    label: str
    generator: str
    ks: tuple[int, ...]
    results: list[ItemResult] = field(default_factory=list)

    def outcomes(self) -> Counter:
        return Counter(r.outcome for r in self.results)

    def summary(self, n_boot: int = 2000, seed: int = 0) -> dict[str, dict]:
        """metric -> {mean, lo, hi, n}. Bootstrap resamples questions (95% percentile CI)."""
        cols: dict[str, list[float]] = {}
        for r in self.results:
            for k, v in r.metrics.items():
                cols.setdefault(k, []).append(v)
        out: dict[str, dict] = {}
        for name, vals in cols.items():
            lo, hi = bootstrap_ci(vals, n_boot, seed) if (name != "latency_ms" and n_boot) else (float("nan"),) * 2
            out[name] = {"mean": float(np.mean(vals)), "lo": lo, "hi": hi, "n": len(vals)}
        acc = cols.get("answer_accuracy", [])
        abst = cols.get("abstention_accuracy", [])
        if acc and abst:
            mean, lo, hi = balanced_ci(acc, abst, n_boot, seed) if n_boot else (0.5 * (np.mean(acc) + np.mean(abst)),) + (float("nan"),) * 2
            out["balanced_score"] = {"mean": mean, "lo": lo, "hi": hi, "n": len(acc) + len(abst)}
        lat = cols.get("latency_ms", [])
        if lat:
            out["latency_p50_ms"] = {"mean": float(np.percentile(lat, 50)), "lo": float("nan"), "hi": float("nan"), "n": len(lat)}
            out["latency_p95_ms"] = {"mean": float(np.percentile(lat, 95)), "lo": float("nan"), "hi": float("nan"), "n": len(lat)}
        return out

    def by_type(self) -> dict[str, "EvalRun"]:
        groups: dict[str, EvalRun] = {}
        for r in self.results:
            g = groups.setdefault(r.item.type, EvalRun(self.label, self.generator, self.ks))
            g.results.append(r)
        return groups

    def rows(self) -> list[dict]:
        return [r.to_row() for r in self.results]


def _classify(item: EvalItem, ans: Answer, correct: bool) -> str:
    if not item.answerable:
        return "correct_abstention" if ans.abstained else "over_answer"
    seen = evidence_in_context(ans, item.evidence)
    if ans.abstained:
        return "false_refusal_evidence_present" if seen else "false_refusal_retrieval"
    if correct:
        return "correct"
    return "wrong_answer_generation" if seen else "wrong_answer_retrieval"


def evaluate(pipeline: RAGPipeline, items: Sequence[EvalItem], ks: Sequence[int] = DEFAULT_KS,
             judge: Callable | None = None, support_threshold: float = DEFAULT_THRESHOLD,
             label: str | None = None) -> EvalRun:
    ks = tuple(sorted(ks))
    run = EvalRun(label or pipeline.config.label(), pipeline.generator.name, ks)
    for item in items:
        ans = pipeline.ask(item.question)
        m: dict[str, float] = {"latency_ms": ans.latency_ms}
        correct = False

        if item.answerable:
            ranked = pipeline.retrieve(item.question, max(ks))
            n_rel = sum(any(phrase_in(c.text, p) for p in item.evidence) for c in pipeline.chunks)
            m.update(retrieval_metrics(ranked, item.evidence, ks, n_rel))
            recall = fact_recall(ans.text, item.facts) if not ans.abstained else 0.0
            correct = (not ans.abstained) and recall == 1.0
            m["answer_accuracy"] = float(correct)
            m["fact_recall"] = recall
            m["false_refusal_rate"] = float(ans.abstained)
            if not ans.abstained:
                m["token_f1"] = token_f1(ans.text, item.answer)
                m["citation_precision"], m["citation_recall"] = citation_scores(ans, item.evidence)
        else:
            m["abstention_accuracy"] = float(ans.abstained)

        faith = judgement = None
        if not ans.abstained:
            faith = score_answer(ans.text, [h.chunk.text for h in ans.contexts], support_threshold)
            m["faithfulness"] = faith.score
            m["unfaithful_rate"] = float(faith.unfaithful)
            if judge is not None:
                judgement = judge(item.question, [h.chunk.text for h in ans.contexts], item.answer, ans.text)
                if judgement is not None:
                    m["judge_faithful"] = float(judgement["faithful"])
                    if item.answerable:
                        m["judge_correct"] = float(judgement["correct"])
        m["hallucination_rate"] = float((not ans.abstained) and ((not item.answerable) or faith.unfaithful))
        run.results.append(ItemResult(item, ans, m, _classify(item, ans, correct), faith, judgement))
    return run
