"""Quality gates: fail a CI run when a metric regresses past a floor (or a ceiling for error rates)."""
from __future__ import annotations

import json
from pathlib import Path

# metric -> direction. Floors for "higher is better" metrics, ceilings for error rates.
HIGHER_IS_BETTER = ("hit@5", "recall@5", "mrr", "answer_accuracy", "abstention_accuracy", "balanced_score")
LOWER_IS_BETTER = ("hallucination_rate", "unfaithful_rate")


def make_gates(summary: dict, margin: float = 0.08) -> dict:
    """Set each gate `margin` worse than the current in-sample value, so noise doesn't fail CI but real regressions do."""
    gates: dict = {}
    for k in HIGHER_IS_BETTER:
        if k in summary:
            gates[k] = {"min": round(max(summary[k]["mean"] - margin, 0.0), 2)}
    for k in LOWER_IS_BETTER:
        if k in summary:
            gates[k] = {"max": round(min(summary[k]["mean"] + margin, 1.0), 2)}
    return gates


def check_gates(summary: dict, gates: dict) -> list[str]:
    """Return human-readable failures (empty list = all gates pass)."""
    failures = []
    for metric, bound in gates.items():
        if metric not in summary:
            failures.append(f"{metric}: not measured")
            continue
        value = summary[metric]["mean"]
        if "min" in bound and value < bound["min"]:
            failures.append(f"{metric} = {value:.3f} is below the floor {bound['min']}")
        if "max" in bound and value > bound["max"]:
            failures.append(f"{metric} = {value:.3f} is above the ceiling {bound['max']}")
    return failures


def load_gates(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
