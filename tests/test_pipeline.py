import io
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _common import (
    extend_training_stories, file_hash, load_checkpoint, load_tokens, make_model,
    sample_batch, split_stories,
)
from download_data import download_stories
from smallGPT.layers.attantion import CausalSelfAttantion, MultiHeadCausalSelfAttantion


class DataTests(unittest.TestCase):
    def test_extension_preserves_held_out_data_and_deduplicates_train(self):
        previous = {"train": ["old"], "validation": ["val"], "test": ["test"]}
        expanded = extend_training_stories(["new", "val", "old", "test", "new", ""], previous)
        self.assertEqual(expanded, {"train": ["old", "new"], "validation": ["val"], "test": ["test"]})
        self.assertEqual(previous["train"], ["old"])
        with self.assertRaises(ValueError):
            extend_training_stories([], {"train": ["val"], "validation": ["val"], "test": ["test"]})

    def test_split_removes_duplicates_without_leakage_or_global_rng_changes(self):
        stories = [f"Story {i}" for i in range(20)] + ["Story 0", "Story 1", ""]
        before = random.getstate()
        splits = split_stories(stories, 0.2, 0.2, 42)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(splits, split_stories(stories, 0.2, 0.2, 42))
        train, validation, test = [set(splits[name]) for name in ("train", "validation", "test")]
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertEqual(len(train | validation | test), 20)

    def test_sample_batch_has_shifted_targets_and_uses_own_generator(self):
        tokens = torch.arange(30)
        before = torch.get_rng_state().clone()
        generator = torch.Generator().manual_seed(42)
        x, y = sample_batch(tokens, 4, 8, generator, "cpu")
        self.assertEqual(x.shape, (4, 8))
        torch.testing.assert_close(y, x + 1)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            sample_batch(torch.arange(8), 1, 8, generator, "cpu")

    def test_download_reads_whole_stories_across_utf8_chunk_boundaries(self):
        first = "α" * 32767 + "🙂"
        source = (first + "<|endoftext|>second<|endoftext|>unfinished").encode("utf-8")
        with patch("download_data.urlopen", return_value=io.BytesIO(source)):
            self.assertEqual(download_stories("https://example.com/data", 2), [first, "second"])
        with patch("download_data.urlopen", return_value=io.BytesIO(source)):
            with self.assertRaises(RuntimeError):
                download_stories("https://example.com/data", 3)


class TrainingPipelineTests(unittest.TestCase):
    def run_script(self, name, *arguments):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / name), *map(str, arguments)],
            cwd=ROOT, text=True, capture_output=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_prepare_train_exact_resume_evaluate_and_generate(self):
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            raw = temporary / "raw.jsonl"
            texts = [f"Story {i}: A little fox found a red ball. The fox played with a friend." for i in range(20)]
            raw.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
            data = temporary / "processed"
            self.run_script("prepare_data.py", "--input", raw, "--output-dir", data,
                            "--vocab-size", 100, "--tokenizer-stories", 0,
                            "--validation-fraction", 0.2, "--test-fraction", 0.2)
            for split in ("train", "validation", "test"):
                tokens = load_tokens(data, split)
                metadata = json.loads((data / "metadata.json").read_text())
                self.assertEqual(len(tokens), metadata["splits"][split]["tokens"])
                self.assertEqual((tokens == 1).sum().item(), metadata["splits"][split]["stories"])
                self.assertEqual((tokens == 2).sum().item(), metadata["splits"][split]["stories"])
            expanded_raw = temporary / "expanded.jsonl"
            expanded_raw.write_text(raw.read_text() + json.dumps({"text": "A new story about a friendly fox."}) + "\n",
                                    encoding="utf-8")
            expanded_data = temporary / "expanded"
            self.run_script("prepare_data.py", "--input", expanded_raw, "--extend-from", data,
                            "--output-dir", expanded_data)
            for filename in ("tokenizer.json", "validation.jsonl", "validation.pt", "test.jsonl", "test.pt"):
                self.assertEqual(file_hash(data / filename), file_hash(expanded_data / filename))
            expanded_metadata = json.loads((expanded_data / "metadata.json").read_text())
            self.assertEqual(expanded_metadata["extended_from"]["added_train_stories"], 1)
            new_vocabulary = temporary / "new_vocabulary"
            self.run_script("prepare_data.py", "--splits-from", expanded_data,
                            "--output-dir", new_vocabulary, "--vocab-size", 120,
                            "--tokenizer-stories", 0)
            new_metadata = json.loads((new_vocabulary / "metadata.json").read_text())
            self.assertEqual(new_metadata["vocab_size"], 120)
            self.assertEqual(new_metadata["raw_sha256"], expanded_metadata["raw_sha256"])
            self.assertEqual(new_metadata["tokenizer_training_stories"], 13)
            for split in ("train", "validation", "test"):
                self.assertEqual((expanded_data / f"{split}.jsonl").read_bytes(),
                                 (new_vocabulary / f"{split}.jsonl").read_bytes())
                self.assertEqual(len(load_tokens(new_vocabulary, split)), new_metadata["splits"][split]["tokens"])
            options = ["--data-dir", data, "--device", "cpu", "--embedding-dim", 8,
                       "--context-length", 8, "--batch-size", 2, "--steps", 6,
                       "--warmup-steps", 2, "--eval-every", 3, "--eval-batches", 2,
                       "--accumulation", 2, "--cpu-threads", 1]
            continuous = temporary / "continuous"
            interrupted = temporary / "interrupted"
            output = self.run_script("train.py", *options, "--output-dir", continuous)
            update_lines = [line for line in output.splitlines() if line.startswith("update=")]
            self.assertEqual(len(update_lines), 2)
            self.assertTrue(all("train_loss=" in line and "val_loss=" in line for line in update_lines))
            self.run_script("train.py", *options, "--output-dir", interrupted, "--stop-after", 3)
            self.run_script("train.py", "--data-dir", data, "--device", "cpu",
                            "--resume", interrupted / "last.pt", "--output-dir", interrupted,
                            "--cpu-threads", 1)
            complete = load_checkpoint(continuous / "last.pt")
            resumed = load_checkpoint(interrupted / "last.pt")
            self.assertEqual(complete["step"], 6)
            self.assertEqual(resumed["step"], 6)
            self.assertEqual(complete["model_config"]["num_layers"], 4)
            self.assertEqual(complete["model_config"]["num_heads"], 4)
            model = make_model(complete["model_config"], "cpu")
            self.assertIsInstance(model.transformer_block1.attention, MultiHeadCausalSelfAttantion)
            for name, parameter in complete["model"].items():
                torch.testing.assert_close(parameter, resumed["model"][name], rtol=0, atol=0)
            torch.testing.assert_close(complete["sampler_state"], resumed["sampler_state"], rtol=0, atol=0)
            self.assertEqual(complete["scheduler"], resumed["scheduler"])
            evaluation = self.run_script("evaluate.py", "--checkpoint", continuous / "last.pt",
                                         "--data-dir", data, "--device", "cpu")
            self.assertIn(f"tokens={len(load_tokens(data, 'test')) - 1:,}", evaluation)
            self.assertIn("perplexity=", evaluation)
            generated = self.run_script("generate.py", "--checkpoint", continuous / "last.pt",
                                        "--device", "cpu", "--prompt", "A little fox", "--max-new-tokens", 3)
            self.assertTrue(generated.startswith("A little fox"))

    def test_old_model_config_loads_two_blocks(self):
        model = make_model({"vocab_size": 100, "embedding_dim": 8, "context_length": 8}, "cpu")
        self.assertEqual(model.num_layers, 2)
        self.assertEqual(model.num_heads, 1)
        self.assertFalse(hasattr(model, "transformer_block3"))
        self.assertIsInstance(model.transformer_block1.attention, CausalSelfAttantion)

    def test_previous_four_block_config_keeps_single_head(self):
        model = make_model({"vocab_size": 100, "embedding_dim": 8,
                            "context_length": 8, "num_layers": 4}, "cpu")
        self.assertEqual(model.num_layers, 4)
        self.assertEqual(model.num_heads, 1)
        self.assertEqual(model.transformer_block4.attention.num_heads, 1)
        self.assertIsInstance(model.transformer_block4.attention, CausalSelfAttantion)


if __name__ == "__main__":
    unittest.main()
