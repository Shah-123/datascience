"""Evaluation harness: golden set, metrics, faithfulness scoring, ablations and reports."""
from .dataset import EvalItem, check_integrity, load_golden
from .harness import EvalRun, ItemResult, evaluate

__all__ = ["EvalItem", "EvalRun", "ItemResult", "check_integrity", "evaluate", "load_golden"]
