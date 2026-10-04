"""Dependency-free text embeddings.

`HashingEmbedder` maps text to a fixed-size vector with the hashing trick over
word unigrams and bigrams. It is deterministic, needs no model download and is
good enough for routing, retrieval and similarity checks in a local setup.
Swap in a real model by implementing the `Embedder` protocol.
"""

from __future__ import annotations

import re
import zlib
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    me my of on or our please should so than that the their them then there these they this to us was we were
    what when where which who why will with would you your about also any all""".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed and a light plural stem applied."""
    out: list[str] = []
    for tok in tokenize(text):
        if tok in STOPWORDS:
            continue
        out.append(stem(tok))
    return out


_SUFFIXES: tuple[str, ...] = ("ations", "ation", "ments", "ment", "ions", "ion", "ings", "ing", "ed", "ly")
_VOWELS = frozenset("aeiou")


def stem(token: str) -> str:
    """Crude suffix stripper: enough to match "encrypted" with "encryption" and "plans" with "plan"."""
    if token.isdigit() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("es") and not token.endswith(("ses", "ies")) and len(token) > 4:
        token = token[:-1]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 3:
        token = token[:-1]
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            token = token[: -len(suffix)]
            break
    if len(token) > 4 and token[-1] == token[-2] and token[-1] not in _VOWELS and token[-1] not in "sz":
        token = token[:-1]
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an L2-normalised matrix of shape (len(texts), dim)."""
        ...


class HashingEmbedder:
    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    @lru_cache(maxsize=2048)
    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = content_tokens(text)
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for gram in grams:
            h = zlib.crc32(gram.encode("utf-8"))
            sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
            vec[h % self.dim] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        vec.setflags(write=False)
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))
