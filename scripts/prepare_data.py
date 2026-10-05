"""Разделить истории, обучить ручной BPE только на train и сохранить потоки ID."""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch

from _common import (
    ROOT, extend_training_stories, file_hash, read_stories, split_stories, write_stories,
)
from smallGPT.tokenizer.bpe import BPETokeniser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/stories.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--tokenizer-stories", type=int, default=100,
                        help="Train-only sample for slow handwritten BPE; 0 uses all train stories")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    previous = parser.add_mutually_exclusive_group()
    previous.add_argument("--extend-from", type=Path,
                          help="Keep this prepared dataset's tokenizer, validation and test; add stories only to train")
    previous.add_argument("--splits-from", type=Path,
                          help="Keep this dataset's exact story splits, train a new tokenizer and re-encode all splits")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.tokenizer_stories < 0:
        parser.error("--tokenizer-stories must be non-negative")
    if any(args.output_dir.glob("*.pt")) and not args.overwrite:
        parser.error("Prepared data already exists; use --overwrite or another output directory")
    previous_metadata = None
    previous_directory = args.extend_from or args.splits_from
    if previous_directory:
        if args.output_dir.resolve() == previous_directory.resolve():
            parser.error("Reusing prepared data requires a different --output-dir")
        previous_metadata = json.loads((previous_directory / "metadata.json").read_text(encoding="utf-8"))
        previous_tokenizer = previous_directory / "tokenizer.json"
        if file_hash(previous_tokenizer) != previous_metadata["tokenizer_sha256"]:
            raise ValueError("Previous tokenizer differs from metadata")
        for name in ("train", "validation", "test"):
            if file_hash(previous_directory / f"{name}.pt") != previous_metadata["splits"][name]["sha256"]:
                raise ValueError(f"Previous {name} tokens differ from metadata")
        previous_splits = {name: read_stories(previous_directory / f"{name}.jsonl")
                           for name in ("train", "validation", "test")}
        # Пустое расширение также проверяет отсутствие дубликатов и пересечений.
        additions = read_stories(args.input) if args.extend_from else []
        splits = extend_training_stories(additions, previous_splits)
    else:
        splits = split_stories(read_stories(args.input), args.validation_fraction, args.test_fraction, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Splits: { {name: len(texts) for name, texts in splits.items()} }")
    started = time.perf_counter()
    tokenizer_path = args.output_dir / "tokenizer.json"
    if args.extend_from:
        tokenizer = BPETokeniser.load(previous_tokenizer)
        shutil.copyfile(previous_tokenizer, tokenizer_path)
        print(f"Reusing BPE vocabulary={tokenizer.vocab_size}; validation/test unchanged", flush=True)
    else:
        sample = random.Random(args.seed).sample(splits["train"], len(splits["train"]))
        if args.tokenizer_stories:
            sample = sample[:args.tokenizer_stories]
        # Односимвольные последовательности добавляют алфавит train без новых пар.
        alphabet = sorted(set("".join(splits["train"])))
        print(f"Training BPE on {len(sample)} train stories; vocabulary limit={args.vocab_size}", flush=True)
        tokenizer = BPETokeniser().fit(sample + alphabet, args.vocab_size)
        tokenizer.save(tokenizer_path)
    metadata = {
        "seed": args.seed,
        "raw_sha256": (previous_metadata["raw_sha256"] if args.splits_from else file_hash(args.input)),
        "tokenizer_sha256": file_hash(tokenizer_path), "vocab_size": tokenizer.vocab_size,
        "tokenizer_training_stories": (previous_metadata["tokenizer_training_stories"]
                                       if args.extend_from else len(sample)), "splits": {},
        "packing": "BOS + story + EOS; cross-story attention allowed; no padding",
    }
    if args.extend_from:
        metadata["extended_from"] = {
            "metadata_sha256": file_hash(args.extend_from / "metadata.json"),
            "previous_train_stories": len(previous_splits["train"]),
            "added_train_stories": len(splits["train"]) - len(previous_splits["train"]),
        }
    if args.splits_from:
        metadata["splits_from"] = {"metadata_sha256": file_hash(previous_directory / "metadata.json")}
    for name, texts in splits.items():
        if args.extend_from and name != "train":
            for suffix in ("jsonl", "pt"):
                shutil.copyfile(args.extend_from / f"{name}.{suffix}", args.output_dir / f"{name}.{suffix}")
            metadata["splits"][name] = previous_metadata["splits"][name].copy()
            print(f"{name}: kept {len(texts)} stories, {metadata['splits'][name]['tokens']} tokens", flush=True)
            continue
        write_stories(args.output_dir / f"{name}.jsonl", texts)
        ids = []
        for index, text in enumerate(texts, 1):
            ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
            if index % 1000 == 0:
                print(f"{name}: encoded {index}/{len(texts)} stories", flush=True)
        tokens = torch.tensor(ids, dtype=torch.long)
        path = args.output_dir / f"{name}.pt"
        torch.save(tokens, path)
        unknowns = ids.count(tokenizer.unk_id)
        metadata["splits"][name] = {
            "stories": len(texts), "tokens": len(ids), "unknown_tokens": unknowns,
            "unknown_fraction": unknowns / max(1, len(ids)), "sha256": file_hash(path),
        }
        print(f"{name}: {len(ids)} tokens, UNK={unknowns}", flush=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared data in {time.perf_counter() - started:.1f}s: {args.output_dir}")


if __name__ == "__main__":
    main()
