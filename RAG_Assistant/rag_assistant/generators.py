"""Answer generators: turn (question, retrieved passages) into a cited answer - or an abstention.

* ExtractiveGenerator - offline, deterministic. Picks the sentences that cover the question's
  rare terms and abstains when coverage is too low. It can only quote the documents, so it is
  faithful by construction; its weakness is choosing the wrong sentence or abstaining wrongly.
* LLMGenerator - grounded prompting of any ChatClient (Anthropic, ModelScope, OpenAI-compatible).
"""
from __future__ import annotations

import math
import re
import time
from typing import Sequence

from .ingest import Chunk
from .llm import ChatClient, LLMError, LLMRefusal
from .text import content_terms, is_number, split_sentences, tokenize
from .types import REFUSAL_TEXT, Answer, Hit


class TermStats:
    """Inverse document frequencies over the indexed chunks (shared by abstention logic)."""

    def __init__(self, chunks: Sequence[Chunk]):
        self.n = len(chunks)
        self.df: dict[str, int] = {}
        for c in chunks:
            for t in set(content_terms(f"{c.heading}. {c.text}")):
                self.df[t] = self.df.get(t, 0) + 1

    def idf(self, term: str) -> float:
        """Unseen terms get the maximum idf: 'mascot' is not in the corpus, so it must matter."""
        return math.log((self.n + 1) / (self.df.get(term, 0) + 0.5))

# --- answer-type gating ---------------------------------------------------------------------
# A "how much" question needs an amount, a "when" question needs a date, and so on. If none of
# the selected sentences can supply that type, the generator abstains instead of quoting a
# topically-related but non-answering sentence.
_MONTHS = frozenset("january february march april may june july august september october november december".split())
_WEEKDAYS = frozenset("monday tuesday wednesday thursday friday saturday sunday".split())
_QUESTION_TYPES = [
    ("time", re.compile(r"\bwhat time\b|\buntil what time\b|\bhow late\b|\bhow early\b", re.I)),
    ("money", re.compile(r"\bhow much\b|\b(cost|costs|price|fee|fees|fine|stipend|deposit|charge|charged|wage|worth)\b", re.I)),
    ("count", re.compile(r"\bhow (many|long|soon|often|fast)\b|\bwhich week\b|\bwhat (gpa|score|percentage|percent)\b", re.I)),
    ("date", re.compile(r"\bwhen\b|\bdeadline\b|\bby what date\b|\bwhat date\b|\bearliest\b|\blatest\b", re.I)),
]


def expected_answer_type(question: str) -> str | None:
    for kind, pattern in _QUESTION_TYPES:
        if pattern.search(question):
            return kind
    return None


def satisfies_type(sentence: str, kind: str) -> bool:
    raw = sentence.lower()
    toks = tokenize(sentence)
    if kind == "money":
        return "$" in sentence or "%" in sentence or bool(re.search(r"\b(free|waived|no charge)\b", raw))
    if kind == "count":
        return any(is_number(t) for t in toks) or bool(re.search(r"\b(once|twice|every|annual|weekly|daily)\b", raw))
    if kind == "date":
        return any(t in _MONTHS or t in _WEEKDAYS for t in toks)
    if kind == "time":
        return bool(re.search(r"\d\s*(a\.?m|p\.?m)\b|\bnoon\b|\bmidnight\b", raw))
    return True


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, stats: TermStats, min_coverage: float = 0.6, max_sentences: int = 3,
                 min_gain: float = 0.15, rank_bonus: float = 0.05, heading_weight: float = 0.5,
                 min_primary: float = 0.0, type_gate: bool = True, type_bonus: float = 0.15):
        self.stats = stats
        self.type_gate, self.type_bonus = type_gate, type_bonus
        self.min_coverage, self.max_sentences = min_coverage, max_sentences
        self.min_gain, self.rank_bonus, self.heading_weight = min_gain, rank_bonus, heading_weight
        self.min_primary = min_primary  # the best single sentence must cover this much on its own

    def generate(self, question: str, hits: Sequence[Hit]) -> Answer:
        q_terms = set(content_terms(question))
        weights = {t: self.stats.idf(t) for t in q_terms}
        total = sum(weights.values())
        # candidate = (hit_index, position, sentence, {term: credit}); a term found in the sentence
        # itself earns full credit, one found only in the section heading earns `heading_weight`.
        want = expected_answer_type(question) if self.type_gate else None
        cands = []
        for hi, hit in enumerate(hits):
            head = set(content_terms(hit.chunk.heading)) & q_terms
            for pos, sent in enumerate(split_sentences(hit.chunk.text)):
                own = set(content_terms(sent)) & q_terms
                credit = {t: self.heading_weight for t in head}
                credit.update({t: 1.0 for t in own})
                cands.append((hi, pos, sent, credit, want is not None and satisfies_type(sent, want)))

        chosen: list[tuple[int, int, str]] = []
        cov: dict[str, float] = {}
        first_credit: dict[str, float] = {}
        typed = False

        def marginal(c) -> float:
            return sum(weights[t] * max(v - cov.get(t, 0.0), 0.0) for t, v in c[3].items()) / total

        while total > 0 and cands and len(chosen) < self.max_sentences:
            best = max(cands, key=lambda c: (marginal(c) + self.rank_bonus / (1 + c[0]) + self.type_bonus * c[4], len(c[3])))
            if marginal(best) < (self.min_gain if chosen else 1e-9):
                break
            if not chosen:
                first_credit = dict(best[3])
            chosen.append(best[:3])
            typed = typed or best[4]
            for t, v in best[3].items():
                cov[t] = max(cov.get(t, 0.0), v)
            cands.remove(best)

        coverage = sum(weights[t] * v for t, v in cov.items()) / total if total else 0.0
        primary = sum(weights[t] * v for t, v in first_credit.items()) / total if total and first_credit else 0.0
        if not chosen or coverage < self.min_coverage or primary < self.min_primary or (want and not typed):
            return Answer(question, REFUSAL_TEXT, True, [], list(hits), coverage, generator=self.name)
        chosen.sort()
        text = " ".join(f"{sent} [{hi + 1}]" for hi, _, sent in chosen)
        cites = list(dict.fromkeys(hits[hi].chunk.chunk_id for hi, _, _ in chosen))
        return Answer(question, text, False, cites, list(hits), coverage, generator=self.name)


SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided by the user.

Rules:
1. Never use outside knowledge, and never guess. The passages are the only source of truth.
2. If the passages do not contain the information needed to answer the question, reply with exactly: NOT_FOUND
3. Otherwise answer concisely. Copy numbers, dates, amounts and conditions exactly as written.
4. If the answer depends on a condition (for example undergraduate vs graduate, or fall vs spring), state each case.
5. Cite the passages you used as [1], [2] immediately after the claims they support."""

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    blocks = [f"[{i}] ({h.chunk.doc_title} > {h.chunk.heading})\n{h.chunk.text}" for i, h in enumerate(hits, 1)]
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


class LLMGenerator:
    def __init__(self, client: ChatClient, max_tokens: int = 1024):
        self.client, self.max_tokens = client, max_tokens
        self.name = f"llm[{client.name}]"

    def generate(self, question: str, hits: Sequence[Hit]) -> Answer:
        t0 = time.perf_counter()
        try:
            raw = self.client.complete(SYSTEM_PROMPT, build_prompt(question, hits), self.max_tokens)
        except LLMRefusal:
            return Answer(question, REFUSAL_TEXT, True, [], list(hits), 0.0, generator=self.name,
                          meta={"llm_refusal": True})
        except LLMError as exc:
            # Fail closed: an outage must never turn into a made-up answer.
            return Answer(question, REFUSAL_TEXT, True, [], list(hits), 0.0, generator=self.name,
                          meta={"error": str(exc)})
        latency = (time.perf_counter() - t0) * 1000
        if not raw or raw.strip().upper().startswith("NOT_FOUND"):
            return Answer(question, REFUSAL_TEXT, True, [], list(hits), 0.0, latency, self.name, {"raw": raw})
        cited: list[str] = []
        for group in _CITE_RE.findall(raw):
            for num in re.split(r"[,;]\s*", group):
                idx = int(num) - 1
                if 0 <= idx < len(hits) and hits[idx].chunk.chunk_id not in cited:
                    cited.append(hits[idx].chunk.chunk_id)
        return Answer(question, raw, False, cited, list(hits), 1.0, latency, self.name, {"raw": raw})
