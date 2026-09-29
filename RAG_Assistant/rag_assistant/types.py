"""Small shared data types."""
from __future__ import annotations

from dataclasses import dataclass, field

from .ingest import Chunk

REFUSAL_TEXT = "I couldn't find this in the provided documents."


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1-based


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    citations: list[str]  # chunk_ids the answer says it used
    contexts: list[Hit]  # everything the generator was shown, in rank order
    confidence: float
    latency_ms: float = 0.0
    generator: str = ""
    meta: dict = field(default_factory=dict)
