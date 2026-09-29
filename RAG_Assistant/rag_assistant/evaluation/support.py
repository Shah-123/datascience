"""Deterministic faithfulness ("groundedness") scoring.

An answer is split into claims (sentences). A claim is *supported* when some window of 1-3
consecutive context sentences

  1. contains at least ``threshold`` of the claim's content terms,
  2. contains EVERY number in the claim (numeric hallucination is the most common policy-QA failure),
  3. agrees with the claim on negation ("is eligible" vs "is not eligible"). A negation in the
     passage only counts when it sits directly next to terms the claim uses, so quoting the first
     clause of "...charged when the balance is not paid" is not mistaken for a flipped meaning.

This is a cheap lexical proxy for entailment. Its accuracy is measured, not assumed:
see ``data/eval/support_calibration.yaml`` and ``rag_assistant.evaluation.calibration``.
Known blind spot: a claim that attributes the right numbers to the wrong entity when both
appear in adjacent sentences. Use the LLM judge (``--judge``) to cover semantic errors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..text import NEGATIONS, content_terms, is_number, split_sentences

CITATION_RE = re.compile(r"\s*\[\d+(?:\s*[,;]\s*\d+)*\]")
_LEAD_RE = re.compile(r"^\s*(?:yes|no)\b[\s,.:;-]*", re.I)
_HEDGE_RE = re.compile(r"^\s*(?:according to|per|based on|as (?:stated|described|noted|outlined) in)\b[^,]*,\s*", re.I)
DEFAULT_THRESHOLD = 0.75


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def split_claims(answer: str) -> list[str]:
    """Sentences of the answer with citation markers, bullets and yes/no lead-ins removed."""
    claims = []
    for sent in split_sentences(strip_citations(answer)):
        sent = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", sent)
        sent = _HEDGE_RE.sub("", sent)  # "According to the policy, ..." adds no claim of its own
        sent = _LEAD_RE.sub("", sent) if len(content_terms(sent)) > 3 else sent
        if len(content_terms(sent)) >= 2:  # skip fragments like "Yes." or "See below."
            claims.append(sent)
    return claims


@dataclass
class ClaimResult:
    claim: str
    overlap: float
    supported: bool
    reason: str = ""


@dataclass
class Faithfulness:
    claims: list[ClaimResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(c.supported for c in self.claims) / len(self.claims) if self.claims else 1.0

    @property
    def unfaithful(self) -> bool:
        return any(not c.supported for c in self.claims)

    @property
    def unsupported(self) -> list[str]:
        return [c.claim for c in self.claims if not c.supported]


def _windows(passages: Sequence[str], max_span: int = 3):
    """Yield the ordered content-term sequence of every window of 1-3 consecutive sentences."""
    for passage in passages:
        sents = split_sentences(passage)
        for i in range(len(sents)):
            for span in range(1, max_span + 1):
                if i + span > len(sents):
                    break
                yield content_terms(" ".join(sents[i:i + span]), keep_negations=True)


def _scoped_negation(seq: Sequence[str], claim_terms: set[str]) -> bool:
    """Is there a negation in ``seq`` whose immediate neighbours are terms the claim talks about?"""
    for i, tok in enumerate(seq):
        if tok in NEGATIONS:
            neighbours = set(seq[max(i - 1, 0):i]) | set(seq[i + 1:i + 2])
            if neighbours & claim_terms:
                return True
    return False


def score_answer(answer: str, passages: Sequence[str], threshold: float = DEFAULT_THRESHOLD) -> Faithfulness:
    windows = [(seq, set(seq)) for seq in _windows(passages)]
    result = Faithfulness()
    for claim in split_claims(answer):
        terms = set(content_terms(claim, keep_negations=True))
        numbers = {t for t in terms if is_number(t)}
        claim_neg = bool(terms & NEGATIONS)
        best_overlap, reason, supported = 0.0, "no window covers the claim's terms", False
        content = terms - NEGATIONS
        for seq, w_terms in windows:
            overlap = len(terms & w_terms) / len(terms)
            best_overlap = max(best_overlap, overlap)
            if overlap < threshold:
                continue
            if not numbers <= w_terms:
                reason = f"number(s) {sorted(numbers - w_terms)} not found in the supporting passage"
            elif claim_neg != _scoped_negation(seq, content):
                reason = "negation differs from the supporting passage"
            else:
                supported, reason = True, ""
                break
        result.claims.append(ClaimResult(claim, best_overlap, supported, reason))
    return result
