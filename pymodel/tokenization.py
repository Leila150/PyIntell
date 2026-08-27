"""Basic vocabulary and tokenization utilities."""


def tokenizer(text):
    if not isinstance(text, str): raise TypeError("text must be a string")
    return text.split()


def tokenize(text): return tokenizer(text)
def detokenize(tokens): return " ".join(tokens)


def vocab(tokens):
    if isinstance(tokens, str): tokens = tokenizer(tokens)
    if tokens and isinstance(tokens[0], (list, tuple)): tokens = [t for seq in tokens for t in seq]
    result = {}
    for token in tokens:
        if token not in result: result[token] = len(result)
    return result


def build_vocab(tokens): return vocab(tokens)
def update_vocab(vocabulary, tokens):
    result = dict(vocabulary); new = vocab(tokens)
    for token in new:
        if token not in result: result[token] = len(result)
    return result

def merge_vocab(*vocabularies):
    result = {}
    for vocabulary in vocabularies:
        for token in vocabulary:
            if token not in result: result[token] = len(result)
    return result

def reverse_vocab(vocabulary): return {index: token for token, index in vocabulary.items()}
def vocab_size(vocabulary): return len(vocabulary)
def token_id(token, vocabulary, default=None): return vocabulary.get(token, default)
def id_token(index, reverse): return reverse[index]
def token_exists(token, vocabulary): return token in vocabulary

def add_token(vocabulary, token):
    result = dict(vocabulary)
    if token not in result: result[token] = len(result)
    return result

def remove_token(vocabulary, token):
    result = dict(vocabulary); result.pop(token, None); return result

def special_tokens(pad="<PAD>", unk="<UNK>", bos="<BOS>", eos="<EOS>"):
    return {"pad": pad, "unk": unk, "bos": bos, "eos": eos}

def add_special_token(vocabulary, token): return add_token(vocabulary, token)

def encode(text, vocabulary, unknown_token=None):
    tokens = tokenizer(text); unknown_id = vocabulary.get(unknown_token) if unknown_token is not None else None
    return [vocabulary[token] if token in vocabulary else unknown_id for token in tokens]

def decode(ids, vocabulary):
    reverse = vocabulary if vocabulary and all(isinstance(key, int) for key in vocabulary) else reverse_vocab(vocabulary)
    return " ".join(reverse[index] for index in ids)

def encode_batch(texts, vocabulary, unknown_token=None): return [encode(text, vocabulary, unknown_token) for text in texts]
def decode_batch(ids, reverse): return [decode(item, reverse) for item in ids]
def tokenize_batch(texts): return [tokenizer(text) for text in texts]
def detokenize_batch(batch): return [detokenize(tokens) for tokens in batch]

def normalize_text(text): return " ".join(str(text).split())
def clean_text(text): return normalize_text(text)
def split_text(text, separator=None): return str(text).split(separator)
def truncate(text, length): return str(text)[:length]
def pad_sequence(sequence, length, value=0):
    result = list(sequence)[:length]; return result + [value] * max(0, length - len(result))
