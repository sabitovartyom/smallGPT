"""Продолжить prompt сохранённой моделью: temperature, top-k, EOS."""

import argparse
from pathlib import Path

import torch

from _common import load_checkpoint, make_model, select_device, tokenizer_from_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40, help="0 disables filtering")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 0 or args.temperature <= 0 or args.top_k < 0:
        parser.error("Token count and top-k must be non-negative; temperature must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    tokenizer = tokenizer_from_state(checkpoint["tokenizer"])
    model = make_model(checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    context = checkpoint["model_config"]["context_length"]
    ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device)
    continuation = []
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            logits = model(ids[:, -context:])[:, -1, :] / args.temperature
            logits[:, tokenizer.bos_id] = float("-inf")
            logits[:, tokenizer.unk_id] = float("-inf")
            if args.top_k:
                k = min(args.top_k, logits.size(-1) - 2)
                values, indices = torch.topk(logits, k, dim=-1)
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(-1, indices, values)
                logits = filtered
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            if next_id.item() == tokenizer.eos_id:
                break
            continuation.append(next_id.item())
            ids = torch.cat((ids, next_id), dim=1)
    print(args.prompt + tokenizer.decode(continuation))


if __name__ == "__main__":
    main()
