"""Pretrained static word embeddings (GloVe) as a real - if modest - semantic retriever.

Why GloVe: it downloads from a GitHub release (no Hugging Face account/host needed), runs on CPU with
numpy only, and is small (134 MB). Documents/queries are embedded as the idf-weighted mean of their
word vectors, mean-centred, then compared by cosine similarity. It cannot match a transformer
embedding model - for that, plug a sentence-transformers model into ``DenseRetriever`` - but it does
know that "password" and "login secret" are related, which BM25 cannot.

The vector file is cached outside the repository (``~/.cache/rag_assistant`` or ``RAG_CACHE_DIR``);
set ``RAG_GLOVE_PATH`` to use a copy you already have.
"""
from __future__ import annotations

import gzip
import math
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.preprocessing import normalize

from .ingest import Chunk
from .retrievers import Ranking, Retriever, _top_k
from .text import STOPWORDS, tokenize

GLOVE_NAME = "glove-wiki-gigaword-100.gz"
GLOVE_URL = f"https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/{GLOVE_NAME}"
_LOADED: dict[str, tuple[dict[str, int], np.ndarray]] = {}


def cache_dir() -> Path:
    return Path(os.environ.get("RAG_CACHE_DIR") or Path.home() / ".cache" / "rag_assistant")


def glove_path() -> Path:
    return Path(os.environ.get("RAG_GLOVE_PATH") or cache_dir() / GLOVE_NAME)


def glove_available() -> bool:
    return glove_path().is_file()


def ensure_glove(download: bool = True) -> Path:
    path = glove_path()
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(f"GloVe vectors not found at {path}. Run `python -m rag_assistant fetch-embeddings`.")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading GloVe vectors (134 MB) to {path} ...", file=sys.stderr)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(GLOVE_URL, tmp)
    tmp.replace(path)
    return path


def load_glove(path: Path, max_words: int = 100_000) -> tuple[dict[str, int], np.ndarray]:
    """Read the ``max_words`` most frequent words (the file is frequency-ordered)."""
    key = f"{path}:{max_words}"
    if key not in _LOADED:
        words: dict[str, int] = {}
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            next(fh)  # "<n_words> <dim>" header
            for line in fh:
                if len(words) >= max_words:
                    break
                word, _, rest = line.partition(" ")
                words[word] = len(rows)
                rows.append(np.array(rest.split(), dtype=np.float32))
        _LOADED[key] = (words, np.vstack(rows))
    return _LOADED[key]


class GloVeRetriever(Retriever):
    name = "glove"

    def __init__(self, add_heading: bool = True, max_words: int = 100_000):
        self.add_heading, self.max_words = add_heading, max_words

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in tokenize(text) if t not in STOPWORDS]

    def _embed(self, tokens: Sequence[str]) -> np.ndarray:
        vec, total = np.zeros(self.vecs.shape[1], dtype=np.float64), 0.0
        for tok in tokens:
            row = self.words.get(tok)
            if row is not None:
                w = self.idf.get(tok, self.max_idf)  # words absent from the corpus are informative too
                vec += w * self.vecs[row]
                total += w
        return vec / total if total else vec

    def fit(self, chunks: Sequence[Chunk]) -> "GloVeRetriever":
        self.words, self.vecs = load_glove(ensure_glove(), self.max_words)
        docs = [self._tokens(t) for t in self._texts(chunks)]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
        self.max_idf = math.log((n + 1) / 0.5)
        m = np.stack([self._embed(d) for d in docs])
        self.mean = m.mean(axis=0)
        self.matrix = normalize(m - self.mean)
        return self

    def search(self, query: str, k: int) -> Ranking:
        q = normalize((self._embed(self._tokens(query)) - self.mean)[None, :])[0]
        return _top_k(self.matrix @ q, k)
