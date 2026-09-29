"""The RAG pipeline: chunk -> index -> retrieve -> generate."""
from __future__ import annotations

import time
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .config import BEST_CONFIG, CORPUS_DIR
from .generators import ExtractiveGenerator, LLMGenerator, TermStats
from .ingest import Chunk, ChunkConfig, Section, chunk_sections, load_corpus
from .llm import ChatClient
from .retrievers import build_retriever
from .types import Answer, Hit


@dataclass(frozen=True)
class PipelineConfig:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retriever: str = "hybrid"
    top_k: int = 5
    generator: str = "extractive"  # "extractive" | "llm"
    min_coverage: float = 0.6  # extractive abstention threshold
    max_sentences: int = 3
    heading_weight: float = 0.5  # credit for a question term found only in a section heading
    min_primary: float = 0.0  # best single sentence must cover this share of the question terms
    type_gate: bool = True  # abstain if no chosen sentence has the expected answer type ($, date, number...)

    def with_(self, **kw) -> "PipelineConfig":
        return replace(self, **kw)

    def label(self) -> str:
        return f"{self.retriever} | {self.chunk.label()} | k={self.top_k}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        data = dict(data)
        data["chunk"] = ChunkConfig(**data.get("chunk", {}))
        return cls(**data)


def load_best_config(path=BEST_CONFIG) -> PipelineConfig:
    """The configuration selected by `python -m rag_assistant study`, or library defaults if none exists yet."""
    try:
        return PipelineConfig.from_dict(json.loads(open(path, encoding="utf-8").read()))
    except (OSError, ValueError, TypeError):
        return PipelineConfig()


class RAGPipeline:
    def __init__(self, sections: list[Section], config: PipelineConfig | None = None,
                 llm: ChatClient | None = None):
        self.config = config or PipelineConfig()
        self.sections = sections
        self.chunks: list[Chunk] = chunk_sections(sections, self.config.chunk)
        self.retriever = build_retriever(self.config.retriever, self.config.chunk.add_heading).fit(self.chunks)
        self.stats = TermStats(self.chunks)
        if self.config.generator == "llm":
            if llm is None:
                raise ValueError("generator='llm' needs an LLM client (see rag_assistant.llm.make_llm).")
            self.generator = LLMGenerator(llm)
        else:
            self.generator = ExtractiveGenerator(self.stats, self.config.min_coverage, self.config.max_sentences,
                                                  heading_weight=self.config.heading_weight,
                                                  min_primary=self.config.min_primary, type_gate=self.config.type_gate)

    @classmethod
    def from_corpus(cls, corpus: str | Path = CORPUS_DIR, config: PipelineConfig | None = None,
                    llm: ChatClient | None = None) -> "RAGPipeline":
        return cls(load_corpus(corpus), config, llm)

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        ranked = self.retriever.search(question, k or self.config.top_k)
        return [Hit(self.chunks[i], s, r) for r, (i, s) in enumerate(ranked, start=1)]

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        t1 = time.perf_counter()
        ans = self.generator.generate(question, hits)
        t2 = time.perf_counter()
        ans.latency_ms = (t2 - t0) * 1000
        ans.meta.update(retrieval_ms=(t1 - t0) * 1000, generation_ms=(t2 - t1) * 1000)
        return ans
