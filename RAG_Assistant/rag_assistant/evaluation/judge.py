"""Optional LLM-as-judge. Complements the deterministic metrics; never replaces them.

The deterministic scorer cannot see semantic errors (right numbers, wrong entity). A judge model can,
but judges have their own biases, so the harness reports judge metrics *next to* - not instead of -
the lexical ones, and a judge call that fails or returns junk is excluded rather than guessed.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

from ..llm import ChatClient, LLMError

JUDGE_SYSTEM = """You are a strict grader of a question-answering system that must answer only from provided passages.

You receive: the question, the passages the system saw, a reference answer, and the system's answer.
Grade two things independently:
- "correct": the system's answer states the same key facts as the reference answer (wording may differ; extra correct detail is fine; a wrong or missing key fact is incorrect).
- "faithful": every factual claim in the system's answer is supported by the passages. Numbers, dates, amounts and conditions must match the passages exactly. Claims that come from outside knowledge are NOT faithful.

Reply with ONLY a JSON object: {"correct": true|false, "faithful": true|false, "unsupported_claims": ["..."]}"""


def build_judge_prompt(question: str, passages: Sequence[str], reference: str, candidate: str) -> str:
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return (f"Question: {question}\n\nPassages:\n{ctx}\n\nReference answer: {reference or '(none - the question is unanswerable)'}"
            f"\n\nSystem answer: {candidate}")


def parse_judgement(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data.get("correct"), bool) or not isinstance(data.get("faithful"), bool):
        return None
    claims = data.get("unsupported_claims")
    data["unsupported_claims"] = [str(c) for c in claims] if isinstance(claims, list) else []
    return data


class LLMJudge:
    def __init__(self, client: ChatClient):
        self.client = client
        self.name = f"judge[{client.name}]"

    def __call__(self, question: str, passages: Sequence[str], reference: str, candidate: str) -> dict | None:
        try:
            raw = self.client.complete(JUDGE_SYSTEM, build_judge_prompt(question, passages, reference, candidate), 2048)
        except LLMError:
            return None
        return parse_judgement(raw)
