import pytest

from rag_assistant import PipelineConfig, RAGPipeline
from rag_assistant.config import CORPUS_DIR
from rag_assistant.evaluation import load_golden
from rag_assistant.ingest import load_corpus


@pytest.fixture(scope="session")
def sections():
    return load_corpus(CORPUS_DIR)


@pytest.fixture(scope="session")
def items():
    return load_golden(split="all")


@pytest.fixture(scope="session")
def pipeline(sections):
    # BM25 only: tests must not depend on downloaded embedding weights.
    return RAGPipeline(sections, PipelineConfig(retriever="bm25", top_k=5, min_coverage=0.5))
