"""Golden-set loading and integrity checking."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import GOLDEN_SET
from ..ingest import Section
from ..text import contains_phrase

TYPES = ("factual", "paraphrase", "conditional", "multi_hop", "unanswerable_near", "unanswerable_off")


@dataclass(frozen=True)
class EvalItem:
    id: str
    type: str
    question: str
    answer: str
    facts: tuple[tuple[str, ...], ...]  # every fact must be present; each fact = acceptable spellings
    evidence: tuple[str, ...]
    split: str

    @property
    def answerable(self) -> bool:
        return not self.type.startswith("unanswerable")


def load_golden(path: str | Path = GOLDEN_SET, split: str | None = None) -> list[EvalItem]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    seen: dict[str, int] = defaultdict(int)
    items: list[EvalItem] = []
    for r in raw:
        auto = "dev" if seen[r["type"]] % 2 == 0 else "test"  # alternate within each type
        seen[r["type"]] += 1
        items.append(EvalItem(
            id=r["id"], type=r["type"], question=r["question"], answer=r.get("answer", ""),
            facts=tuple(tuple(f) for f in r.get("facts", [])),
            evidence=tuple(r.get("evidence", [])), split=r.get("split", auto),
        ))
    return [i for i in items if split in (None, "all", i.split)]


def check_integrity(items: list[EvalItem], sections: list[Section]) -> list[str]:
    """Return a list of problems (empty = the golden set is consistent with the corpus)."""
    problems: list[str] = []
    ids = [i.id for i in items]
    problems += [f"duplicate id: {i}" for i in {x for x in ids if ids.count(x) > 1}]
    for it in items:
        if it.type not in TYPES:
            problems.append(f"{it.id}: unknown type {it.type!r}")
            continue
        if it.answerable:
            if not (it.answer and it.facts and it.evidence):
                problems.append(f"{it.id}: answerable items need answer, facts and evidence")
        elif it.facts or it.evidence or it.answer:
            problems.append(f"{it.id}: unanswerable items must have no answer, facts or evidence")

        evidence_text = ""
        for phrase in it.evidence:
            total = sum(_count(s.text, phrase) for s in sections)
            if total != 1:
                problems.append(f"{it.id}: evidence {phrase!r} occurs {total} times in the corpus (need exactly 1)")
                continue
            evidence_text += " " + next(s.text for s in sections if contains_phrase(s.text, phrase))
        for fact in it.facts:
            if not any(contains_phrase(evidence_text, alt) for alt in fact):
                problems.append(f"{it.id}: fact {fact} is not derivable from its evidence sections")
    for split in ("dev", "test"):
        missing = set(TYPES) - {i.type for i in items if i.split == split}
        if missing:
            problems.append(f"split {split!r} has no items of type(s) {sorted(missing)}")
    return problems


def _count(text: str, phrase: str) -> int:
    from ..text import normalize_for_match
    return normalize_for_match(text).count(normalize_for_match(phrase))
