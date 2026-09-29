"""Text utilities shared by indexing, retrieval, generation and evaluation.

Everything that decides "do these two strings mean the same thing" lives here, so the
retriever, the generator and the evaluation harness cannot drift apart.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from as into onto over under
    about is are was were be been being am do does did done have has had having can could may
    might must shall should will would i me my we our you your he she it its they them their
    this that these those there here what which who whom whose when where why how much many
    long often tell please any all each every both either neither some such more most other
    another up down out off again further once only own same s t don doesn didn hru halcyon
    ridge university not no nor so than too very just also yes
    """.split()
)

# "no" is deliberately absent: as a determiner ("No undergraduate may...") it is not a predicate negation.
NEGATIONS = frozenset({"not", "never", "cannot", "without"})

NUMBER_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
}

_TOKEN_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?|[a-z]+(?:'[a-z]+)?")
_NUMBER_RE = re.compile(r"^\$?\d[\d,]*(?:\.\d+)?%?$")
_STEMMER = snowballstemmer.stemmer("english")


def _canon_number(tok: str) -> str:
    return tok.replace("$", "").replace(",", "").rstrip("%")


def tokenize(text: str) -> list[str]:
    """Lowercased word/number tokens; punctuation, currency and thousands separators removed.

    ``$1,150`` -> ``1150``, ``60%`` -> ``60``, ``doesn't`` -> ``not`` (negation kept explicit),
    number words two..twelve/twenty -> digits.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\b([ap])\.m\.?", r"\1m", text)  # "8 a.m." -> "8 am"
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        if tok[0].isdigit() or tok[0] == "$":
            out.append(_canon_number(tok))
        elif tok.endswith("n't"):
            out.append("not")
        else:
            tok = tok.split("'")[0]
            out.append(NUMBER_WORDS.get(tok, tok))
    return out


def is_number(tok: str) -> bool:
    return bool(_NUMBER_RE.match(tok))


@lru_cache(maxsize=50_000)
def stem(tok: str) -> str:
    return tok if is_number(tok) else _STEMMER.stemWord(tok)


def content_terms(text: str, keep_negations: bool = False) -> list[str]:
    """Stemmed non-stopword terms (numbers kept). Used for retrieval and support scoring."""
    out = []
    for tok in tokenize(text):
        if tok in STOPWORDS and not (keep_negations and tok in NEGATIONS):
            continue
        out.append(stem(tok))
    return out


def analyze(text: str) -> list[str]:
    """Index-time analyzer: content terms plus adjacent-pair bigrams (for TF-IDF/LSA)."""
    terms = content_terms(text)
    return terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:])]


def normalize_for_match(text: str) -> str:
    """Canonical form used to test 'does this text contain that phrase'."""
    return " " + " ".join(tokenize(text)) + " "


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize_for_match(phrase) in normalize_for_match(text)


# --- sentence splitting -------------------------------------------------------------------

_PROTECT = [
    (re.compile(r"\b([ap])\.m\."), "\\1\u2024m\u2024"),
    (re.compile(r"\b(e)\.(g)\."), "\\1\u2024\\2\u2024"),
    (re.compile(r"\b(i)\.(e)\."), "\\1\u2024\\2\u2024"),
    (re.compile(r"\b(U)\.(S)\."), "\\1\u2024\\2\u2024"),
    (re.compile(r"\b(vs|etc|approx)\."), "\\1\u2024"),
]
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\"(\[])")


def split_sentences(text: str) -> list[str]:
    """Split on newlines first (paragraphs, bullets, linearized table rows), then on sentence ends."""
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for pat, rep in _PROTECT:
            line = pat.sub(rep, line)
        for part in _SPLIT_RE.split(line):
            part = part.replace("\u2024", ".").strip()
            if part:
                sentences.append(part)
    return sentences


def strip_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`]{1,3}", "", text)
    return text.strip()


def slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text or "section"
