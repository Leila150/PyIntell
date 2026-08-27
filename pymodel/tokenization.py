"""Basic vocabulary and tokenization utilities."""


def tokenizer(text):
    """Tokenize text using whitespace splitting."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.split()


def vocab(tokens):
    """Create a token-to-id vocabulary from tokens or token sequences."""
    if isinstance(tokens, str):
        tokens = tokenizer(tokens)
    if tokens and isinstance(tokens[0], (list, tuple)):
        tokens = [token for sequence in tokens for token in sequence]
    result = {}
    for token in tokens:
        if token not in result:
            result[token] = len(result)
    return result


def reverse_vocab(vocabulary):
    """Reverse a token-to-id vocabulary into an id-to-token vocabulary."""
    return {index: token for token, index in vocabulary.items()}


def encode(text, vocabulary, unknown_token=None):
    """Encode text into token IDs."""
    tokens = tokenizer(text)
    unknown_id = vocabulary.get(unknown_token) if unknown_token is not None else None
    return [vocabulary[token] if token in vocabulary else unknown_id for token in tokens]


def decode(ids, vocabulary):
    """Decode token IDs using an id-to-token vocabulary or token-to-id vocabulary."""
    if vocabulary and all(isinstance(key, int) for key in vocabulary):
        reverse = vocabulary
    else:
        reverse = reverse_vocab(vocabulary)
    return " ".join(reverse[index] for index in ids)
