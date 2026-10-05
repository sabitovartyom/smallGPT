"""Отдельная оценка checkpoint; test не используется training loop."""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import json

from _common import ROOT, file_hash, load_checkpoint, load_tokens, make_model, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(2)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    metadata_path = args.data_dir / "metadata.json"
    if file_hash(metadata_path) != checkpoint["data_fingerprint"]:
        raise ValueError("Evaluation requires the same prepared dataset as the checkpoint")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if file_hash(args.data_dir / f"{args.split}.pt") != metadata["splits"][args.split]["sha256"]:
        raise ValueError("Evaluation tokens differ from prepared metadata")
    model = make_model(checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokens = load_tokens(args.data_dir, args.split)
    context = checkpoint["model_config"]["context_length"]
    if len(tokens) < 2:
        raise ValueError("Evaluation requires at least two tokens")
    if tokens.min() < 0 or tokens.max() >= checkpoint["model_config"]["vocab_size"]:
        raise ValueError("Evaluation token IDs are outside the checkpoint vocabulary")
    total_loss = 0.0
    total_tokens = 0
    # Последняя короткая часть тоже учитывается. Каждый target считается один раз.
    with torch.inference_mode():
        full_windows = (len(tokens) - 1) // context
        for offset in range(0, full_windows, args.batch_size):
            starts = torch.arange(offset, min(offset + args.batch_size, full_windows)) * context
            windows = tokens[starts[:, None] + torch.arange(context + 1)].to(device)
            logits = model(windows[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), windows[:, 1:].reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += windows.size(0) * context
        tail = tokens[full_windows * context:].to(device)
        if len(tail) > 1:
            logits = model(tail[:-1].unsqueeze(0))
            total_loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), tail[1:], reduction="sum").item()
            total_tokens += len(tail) - 1
    average = total_loss / total_tokens
    print(f"{args.split}: tokens={total_tokens:,}, loss={average:.4f}, perplexity={math.exp(average):.2f}")


if __name__ == "__main__":
    main()
