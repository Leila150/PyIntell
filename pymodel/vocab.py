"""Vocabulary management for large-scale pymodel datasets.

This module keeps a forward vocabulary (token -> integer ID) and its exact
reverse vocabulary (integer ID -> token) together so they can never drift
apart during dataset preparation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class Vocabulary:
    """Bidirectional token vocabulary.

    Parameters
    ----------
    tokens:
        Optional iterable of tokens. Tokens are assigned IDs in first-seen
        order unless an existing mapping is supplied with ``vocabulary``.
    vocabulary:
        Optional token -> ID mapping to restore an existing vocabulary.
    """

    def __init__(
        self,
        tokens: Iterable[str] | None = None,
        vocabulary: Mapping[str, int] | None = None,
    ) -> None:
        if vocabulary is not None and tokens is not None:
            raise ValueError("provide either tokens or vocabulary, not both")

        if vocabulary is not None:
            self.vocab = self._validate_vocabulary(vocabulary)
        else:
            self.vocab = {}
            if tokens is not None:
                self.add_many(tokens)

        self.reverse_vocab = self._build_reverse(self.vocab)

    @staticmethod
    def _validate_vocabulary(vocabulary: Mapping[str, int]) -> dict[str, int]:
        result = dict(vocabulary)

        for token, index in result.items():
            if not isinstance(token, str):
                raise TypeError("vocabulary tokens must be strings")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError("vocabulary IDs must be non-negative integers")

        ids = list(result.values())
        if len(ids) != len(set(ids)):
            raise ValueError("vocabulary IDs must be unique")

        if set(ids) != set(range(len(ids))):
            raise ValueError("vocabulary IDs must be contiguous starting at 0")

        return result

    @staticmethod
    def _build_reverse(vocabulary: Mapping[str, int]) -> dict[int, str]:
        return {index: token for token, index in vocabulary.items()}

    def add(self, token: str) -> int:
        """Add one token and return its ID."""
        if not isinstance(token, str):
            raise TypeError("token must be a string")

        if token not in self.vocab:
            index = len(self.vocab)
            self.vocab[token] = index
            self.reverse_vocab[index] = token

        return self.vocab[token]

    def add_many(self, tokens: Iterable[str]) -> None:
        """Add tokens in first-seen order."""
        for token in tokens:
            self.add(token)

    def add_special_tokens(self, tokens: Iterable[str]) -> None:
        """Add special tokens in the supplied order."""
        self.add_many(tokens)

    def contains(self, token: str) -> bool:
        return token in self.vocab

    def token_id(self, token: str, default: Any = None) -> Any:
        return self.vocab.get(token, default)

    def id_token(self, index: int, default: Any = None) -> Any:
        return self.reverse_vocab.get(index, default)

    def encode(self, tokens: Iterable[str], unknown_id: int | None = None) -> list[int | None]:
        """Convert tokens to IDs without modifying the vocabulary."""
        return [self.vocab.get(token, unknown_id) for token in tokens]

    def decode(self, ids: Iterable[int], unknown_token: str | None = None) -> list[str | None]:
        """Convert IDs to tokens without modifying the vocabulary."""
        return [self.reverse_vocab.get(index, unknown_token) for index in ids]

    def size(self) -> int:
        return len(self.vocab)

    def copy(self) -> "Vocabulary":
        return Vocabulary(vocabulary=self.vocab)

    def as_dict(self) -> dict[str, int]:
        return dict(self.vocab)

    def as_reverse_dict(self) -> dict[int, str]:
        return dict(self.reverse_vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, token: str) -> bool:
        return token in self.vocab

    def __getitem__(self, token: str) -> int:
        return self.vocab[token]


def build_vocab(tokens: Iterable[str]) -> dict[str, int]:
    """Build a token -> ID vocabulary in first-seen order."""
    return Vocabulary(tokens=tokens).as_dict()


def reverse_vocab(vocabulary: Mapping[str, int]) -> dict[int, str]:
    """Build the exact ID -> token reverse vocabulary."""
    checked = Vocabulary(vocabulary=vocabulary)
    return checked.as_reverse_dict()


def build_vocabs(tokens: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build both forward and reverse vocabularies from one token stream."""
    vocabulary = Vocabulary(tokens=tokens)
    return vocabulary.as_dict(), vocabulary.as_reverse_dict()


def validate_vocab(vocabulary: Mapping[str, int], reverse: Mapping[int, str] | None = None) -> bool:
    """Validate a vocabulary and optionally verify its reverse mapping."""
    checked = Vocabulary(vocabulary=vocabulary)
    if reverse is not None and dict(reverse) != checked.reverse_vocab:
        raise ValueError("reverse vocabulary does not match vocabulary")
    return True


__all__ = [
    "Vocabulary",
    "build_vocab",
    "reverse_vocab",
    "build_vocabs",
    "validate_vocab",
]
