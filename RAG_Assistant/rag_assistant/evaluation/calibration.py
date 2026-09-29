"""Evaluate the evaluator: how accurate is the faithfulness scorer against hand labels?

Also provides a *fault-injection* sensitivity check: corrupt a known fraction of a pipeline's answers
(altered number, invented sentence, flipped negation) and confirm the harness flags them. If a harness
cannot see faults we planted on purpose, its "0% hallucination" is meaningless.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import CALIBRATION_SET
from ..pipeline import RAGPipeline
from ..types import Answer, Hit
from .dataset import EvalItem
from .harness import EvalRun, evaluate
from .support import DEFAULT_THRESHOLD, score_answer


@dataclass
class CalibrationResult:
    threshold: float
    n: int
    accuracy: float
    unfaithful_precision: float
    unfaithful_recall: float
    false_alarms: list[str] = field(default_factory=list)  # faithful answers flagged unfaithful
    misses: list[str] = field(default_factory=list)  # unfaithful answers passed as faithful
    unexpected: list[str] = field(default_factory=list)  # errors NOT marked expected_miss


def calibrate(path: str | Path = CALIBRATION_SET, threshold: float = DEFAULT_THRESHOLD) -> CalibrationResult:
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tp = fp = fn = tn = 0
    false_alarms, misses, unexpected = [], [], []
    for r in rows:
        predicted_unfaithful = score_answer(r["answer"], r["context"], threshold).unfaithful
        actually_unfaithful = r["label"] == "unfaithful"
        if predicted_unfaithful and actually_unfaithful:
            tp += 1
        elif predicted_unfaithful:
            fp += 1
            false_alarms.append(f"{r['id']} ({r['category']})")
        elif actually_unfaithful:
            fn += 1
            misses.append(f"{r['id']} ({r['category']})")
        else:
            tn += 1
        if predicted_unfaithful != actually_unfaithful and not r.get("expected_miss"):
            unexpected.append(r["id"])
    n = tp + fp + fn + tn
    return CalibrationResult(
        threshold, n, (tp + tn) / n,
        tp / (tp + fp) if tp + fp else 1.0, tp / (tp + fn) if tp + fn else 1.0,
        false_alarms, misses, unexpected,
    )


# ---------------------------------------------------------------------------- fault injection

_NUM = re.compile(r"\d+(?:\.\d+)?")
FAULT_KINDS = ("number", "invented", "negation")
_INVENTED = "Students must also obtain written approval from the Dean of Students within 48 hours."


def _mutate(text: str, kind: str) -> str | None:
    """Return a corrupted copy of ``text`` or None if this kind cannot be applied."""
    if kind == "number":
        m = _NUM.search(text)
        if not m:
            return None
        old = m.group(0)
        new = str(int(old) + 7) if old.isdigit() else f"{float(old) + 0.4:.1f}"
        return text[:m.start()] + new + text[m.end():]
    if kind == "invented":
        return f"{text} {_INVENTED}"
    if kind == "negation":
        for pat, rep in ((r"\bis not\b", "is"), (r"\bare not\b", "are"), (r"\bmay\b", "may not"),
                         (r"\bmust\b", "need not"), (r"\bis\b", "is not"), (r"\bare\b", "are not")):
            if re.search(pat, text):
                return re.sub(pat, rep, text, count=1)
    return None


class FaultInjector:
    """Wraps a generator and corrupts a deterministic fraction of its non-abstaining answers."""

    def __init__(self, base, rate: float = 0.5, seed: int = 0):
        self.base, self.rate, self.seed = base, rate, seed
        self.name = f"fault-injected[{base.name}, rate={rate}]"

    def generate(self, question: str, hits) -> Answer:
        ans = self.base.generate(question, hits)
        if ans.abstained:
            return ans
        h = zlib.crc32(f"{self.seed}:{question}".encode())
        if (h % 1000) / 1000 >= self.rate:
            return ans
        kind = FAULT_KINDS[h % len(FAULT_KINDS)]
        mutated = _mutate(ans.text, kind)
        if mutated is None:  # e.g. no number / no negatable verb in this answer
            kind, mutated = "invented", _mutate(ans.text, "invented")
        ans.meta["fault"] = kind
        ans.text = mutated
        return ans


@dataclass
class SensitivityResult:
    injected: int
    detected: int
    clean: int
    false_alarms: int
    by_kind: dict[str, tuple[int, int]]  # kind -> (detected, injected)

    @property
    def recall(self) -> float:
        return self.detected / self.injected if self.injected else float("nan")

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarms / self.clean if self.clean else float("nan")


def sensitivity(pipeline: RAGPipeline, items: list[EvalItem], rate: float = 0.5) -> tuple[SensitivityResult, EvalRun]:
    pipeline.generator = FaultInjector(pipeline.generator, rate)
    run = evaluate(pipeline, [i for i in items if i.answerable])
    inj = det = clean = fa = 0
    by_kind: dict[str, list[int]] = {}
    for r in run.results:
        if r.answer.abstained or r.faithfulness is None:
            continue
        fault = r.answer.meta.get("fault")
        if fault:
            inj += 1
            det += r.faithfulness.unfaithful
            d = by_kind.setdefault(fault, [0, 0])
            d[0] += r.faithfulness.unfaithful
            d[1] += 1
        else:
            clean += 1
            fa += r.faithfulness.unfaithful
    return SensitivityResult(inj, det, clean, fa, {k: tuple(v) for k, v in by_kind.items()}), run
