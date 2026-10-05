"""Общие операции pipeline. Архитектура модели остаётся в src/smallGPT."""

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_stories(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line)["text"] for line in stream if line.strip()]


def write_stories(path, stories):
    with Path(path).open("w", encoding="utf-8") as stream:
        for text in stories:
            stream.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_stories(stories, validation_fraction, test_fraction, seed):
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")
    # Точные дубликаты удаляются до split, чтобы не попадать в разные части.
    unique = list(dict.fromkeys(text for text in stories if text.strip()))
    if len(unique) < 3:
        raise ValueError("At least three unique non-empty stories are required")
    random.Random(seed).shuffle(unique)
    n_validation = max(1, int(len(unique) * validation_fraction))
    n_test = max(1, int(len(unique) * test_fraction))
    if n_validation + n_test >= len(unique):
        raise ValueError("Not enough stories for a non-empty training split")
    return {
        "validation": unique[:n_validation],
        "test": unique[n_validation:n_validation + n_test],
        "train": unique[n_validation + n_test:],
    }


def load_tokens(directory, split):
    import torch

    tokens = torch.load(Path(directory) / f"{split}.pt", map_location="cpu", weights_only=True)
    if tokens.ndim != 1 or tokens.dtype != torch.long:
        raise ValueError("Expected a one-dimensional torch.long token stream")
    return tokens


def extend_training_stories(stories, previous_splits):
    """Добавить уникальные истории только в train, сохранив контрольные части."""
    seen = set()
    for texts in previous_splits.values():
        for text in texts:
            if text in seen or not text.strip():
                raise ValueError("Previous splits must be non-empty stories without duplicates or overlap")
            seen.add(text)
    splits = {name: list(texts) for name, texts in previous_splits.items()}
    for text in stories:
        if text.strip() and text not in seen:
            splits["train"].append(text)
            seen.add(text)
    return splits


def sample_batch(tokens, batch_size, context_length, generator, device):
    import torch

    if len(tokens) <= context_length:
        raise ValueError("Token stream must contain at least context_length + 1 tokens")
    starts = torch.randint(len(tokens) - context_length, (batch_size,), generator=generator)
    windows = tokens[starts[:, None] + torch.arange(context_length + 1)]
    # Одно окно [T+1] -> вход [T] и следующий токен для каждой позиции [T].
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


def select_device(requested):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process; use --device cpu for debugging")
    return torch.device(requested)


def make_model(config, device):
    from smallGPT.layers.backbone import Backbone

    config = dict(config)
    # Старые checkpoints были обучены с двумя блоками до появления этой настройки.
    config.setdefault("num_layers", 2)
    # До появления num_heads attention всегда работал с одной головой.
    config.setdefault("num_heads", 1)
    return Backbone(**config).to(device)


def tokenizer_state(tokenizer):
    return {
        "version": 1,
        "tokens": [tokenizer.id_to_token[i] for i in range(tokenizer.vocab_size)],
        "merges": tokenizer.merges,
    }


def tokenizer_from_state(state):
    from smallGPT.tokenizer.bpe import BPETokeniser

    tokenizer = BPETokeniser()
    tokenizer.id_to_token = dict(enumerate(state["tokens"]))
    tokenizer.token_to_id = {token: i for i, token in tokenizer.id_to_token.items()}
    tokenizer.merges = [tuple(pair) for pair in state["merges"]]
    tokenizer._is_fitted = True
    return tokenizer


def load_checkpoint(path, device="cpu"):
    import torch

    # Checkpoints создаются этими скриптами и содержат также Python RNG state.
    return torch.load(path, map_location=device, weights_only=False)


def evaluate_loss(model, tokens, batch_size, context_length, batches, device, seed):
    import torch
    import torch.nn.functional as F

    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        with torch.inference_mode():
            for _ in range(batches):
                x, y = sample_batch(tokens, batch_size, context_length, generator, device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total += loss.item()
    finally:
        model.train(was_training)
    return total / batches
