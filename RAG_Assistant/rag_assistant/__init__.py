"""RAG assistant with a built-in evaluation harness (retrieval quality, faithfulness, abstention)."""
from .ingest import Chunk, ChunkConfig, Section, chunk_sections, load_corpus
from .pipeline import PipelineConfig, RAGPipeline
from .types import REFUSAL_TEXT, Answer, Hit

__all__ = [
    "Answer", "Chunk", "ChunkConfig", "Hit", "PipelineConfig", "RAGPipeline", "REFUSAL_TEXT",
    "Section", "chunk_sections", "load_corpus",
]
__version__ = "0.1.0"
