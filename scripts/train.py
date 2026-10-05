"""Обучение существующего Backbone: FP32, AdamW, validation и checkpoints."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _common import (
    ROOT, evaluate_loss, file_hash, load_checkpoint, load_tokens, make_model,
    sample_batch, select_device, tokenizer_state,
)
from smallGPT.tokenizer.bpe import BPETokeniser


def learning_rate_factor(step, steps, warmup):
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, steps - warmup)))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, model, optimizer, scheduler, step, best_loss,
                    model_config, train_config, tokenizer, data_fingerprint, sampler):
    checkpoint = {
        "version": 1, "step": step, "best_validation_loss": best_loss,
        "model_config": model_config, "train_config": train_config,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "tokenizer": tokenizer_state(tokenizer),
        "data_fingerprint": data_fingerprint, "sampler_state": sampler.get_state(),
        "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    temporary = Path(str(path) + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/debug")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000, help="Optimizer updates, not microbatches")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int, help="Stop at this update and save; original schedule is preserved")
    parser.add_argument("--overfit-batch", action="store_true", help="Learn one fixed training batch for debugging")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        config = checkpoint["train_config"]
        model_config = checkpoint["model_config"]
        print("Resume: model and training settings are restored from the checkpoint", flush=True)
    else:
        config = {key: getattr(args, key) for key in (
            "batch_size", "accumulation", "steps", "lr", "warmup_steps",
            "eval_every", "eval_batches", "seed", "overfit_batch",
        )}
        model_config = {"embedding_dim": args.embedding_dim, "context_length": args.context_length,
                        "num_layers": args.num_layers, "num_heads": args.num_heads}
    if any(config[key] < 1 for key in ("batch_size", "accumulation", "steps", "eval_every", "eval_batches")):
        parser.error("Batch sizes, steps and evaluation intervals must be positive")
    if config["lr"] <= 0 or not 0 <= config["warmup_steps"] < config["steps"]:
        parser.error("LR must be positive; warmup must be in [0, steps)")
    if model_config["embedding_dim"] < 1 or model_config["context_length"] < 1:
        parser.error("Embedding dimension and context length must be positive")
    if model_config.get("num_layers", 2) < 1:
        parser.error("Number of layers must be positive")
    num_heads = model_config.get("num_heads", 1)
    if num_heads < 1 or model_config["embedding_dim"] % num_heads != 0:
        parser.error("Embedding dimension must be divisible by a positive number of heads")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint and (args.output_dir / "last.pt").exists():
        parser.error("Run already exists; choose another --output-dir or use --resume")

    metadata_path = args.data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data_fingerprint = file_hash(metadata_path)
    tokenizer_path = args.data_dir / "tokenizer.json"
    if file_hash(tokenizer_path) != metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from prepared data")
    tokenizer = BPETokeniser.load(tokenizer_path)
    if checkpoint and (checkpoint["data_fingerprint"] != data_fingerprint
                       or checkpoint["tokenizer"] != tokenizer_state(tokenizer)):
        raise ValueError("Resume requires the same prepared data and tokenizer")
    train_tokens = load_tokens(args.data_dir, "train")
    validation_tokens = load_tokens(args.data_dir, "validation")
    for name, tokens in (("train", train_tokens), ("validation", validation_tokens)):
        if file_hash(args.data_dir / f"{name}.pt") != metadata["splits"][name]["sha256"]:
            raise ValueError(f"{name} tokens differ from prepared metadata")
        if len(tokens) <= model_config["context_length"]:
            raise ValueError(f"{name} needs at least context_length + 1 tokens")
        if tokens.min() < 0 or tokens.max() >= tokenizer.vocab_size:
            raise ValueError(f"Invalid token IDs in {name}")
    model_config["vocab_size"] = tokenizer.vocab_size
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = make_model(model_config, device)
    # Матрицы — с weight decay; bias и LayerNorm — без decay.
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": 0.01},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ], lr=config["lr"], betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, config["steps"], config["warmup_steps"])
    )
    sampler = torch.Generator().manual_seed(config["seed"] + 1)
    step = 0
    best_loss = float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        sampler.set_state(checkpoint["sampler_state"])
        random.setstate(checkpoint["rng"]["python"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        if device.type == "cuda" and checkpoint["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        step = checkpoint["step"]
        best_loss = checkpoint["best_validation_loss"]

    context = model_config["context_length"]
    batch_size = config["batch_size"]
    fixed = None
    if config["overfit_batch"]:
        fixed = sample_batch(train_tokens, batch_size, context,
                             torch.Generator().manual_seed(config["seed"] + 2), device)
    print(f"Device={device}; parameters={sum(p.numel() for p in parameters):,}; "
          f"layers={model.num_layers}; heads={model.num_heads}; "
          f"x/y=[{batch_size},{context}]; logits=[{batch_size},{context},{tokenizer.vocab_size}]", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_validation = evaluate_loss(model, validation_tokens, batch_size, context,
                                       config["eval_batches"], device, config["seed"] + 3)
    print(f"Starting at update {step}; validation loss={initial_validation:.4f}", flush=True)
    if not checkpoint:
        (args.output_dir / "config.json").write_text(
            json.dumps({"model": model_config, "training": config}, indent=2), encoding="utf-8"
        )
    limit = min(config["steps"], args.stop_after or config["steps"])
    started = time.perf_counter()
    started_step = step
    interval_train_loss = 0.0
    interval_updates = 0
    model.train()
    while step < limit:
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(config["accumulation"]):
            x, y = fixed if fixed is not None else sample_batch(train_tokens, batch_size, context, sampler, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at update {step + 1}")
            (loss / config["accumulation"]).backward()
            train_loss += loss.detach().item() / config["accumulation"]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step += 1
        interval_train_loss += train_loss
        interval_updates += 1
        row = {"step": step, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"],
               "gradient_norm": grad_norm.item()}
        if step % config["eval_every"] == 0 or step == limit:
            row["mean_train_loss"] = interval_train_loss / interval_updates
            validation = evaluate_loss(model, validation_tokens, batch_size, context,
                                       config["eval_batches"], device, config["seed"] + 3)
            row["validation_loss"] = validation
            improved = validation < best_loss
            best_loss = min(best_loss, validation)
            save_args = (model, optimizer, scheduler, step, best_loss, model_config,
                         config, tokenizer, data_fingerprint, sampler)
            save_checkpoint(args.output_dir / "last.pt", *save_args)
            if improved:
                save_checkpoint(args.output_dir / "best.pt", *save_args)
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if "validation_loss" in row:
            elapsed = time.perf_counter() - started
            rate = (step - started_step) * batch_size * context * config["accumulation"] / max(elapsed, 1e-9)
            print(f"update={step} train_loss={row['mean_train_loss']:.4f} "
                  f"val_loss={row['validation_loss']:.4f} tokens/s={rate:.0f}", flush=True)
            interval_train_loss = 0.0
            interval_updates = 0
    print(f"Finished at update {step}; checkpoints: {args.output_dir}", flush=True)
    if device.type == "cuda":
        print(f"Peak PyTorch memory: allocated={torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB, "
              f"reserved={torch.cuda.max_memory_reserved(device) / 2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
