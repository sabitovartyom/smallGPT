import json
import unittest
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory


BPETokeniser = run_path(
    str(Path(__file__).resolve().parents[1] / "src/smallGPT/tokenizer/bpe.py")
)["BPETokeniser"]


class BPETokeniserTests(unittest.TestCase):
    def test_known_merges_and_vocabulary(self):
        tokenizer = BPETokeniser().fit("abab", 100)
        self.assertEqual(tokenizer.merges, [("a", "b"), ("ab", "ab")])
        self.assertEqual(tokenizer.encode("abab"), [6])
        self.assertEqual(tokenizer.decode([6]), "abab")
        self.assertEqual(tokenizer.vocab_size, 7)
        for token, token_id in tokenizer.token_to_id.items():
            self.assertEqual(tokenizer.id_to_token[token_id], token)

    def test_encode_uses_merge_order_instead_of_longest_match(self):
        tokenizer = BPETokeniser().fit(["bcd", "abc"], 100)
        self.assertEqual(
            tokenizer.encode("abcd"),
            [tokenizer.token_to_id["a"], tokenizer.token_to_id["bcd"]],
        )

    def test_round_trip_and_input_preservation(self):
        texts = ["Привет, мир!", "hello hello", "", "🙂\n"]
        original = texts.copy()
        tokenizer = BPETokeniser().fit(texts, 100)
        self.assertEqual(texts, original)
        for text in texts + ["мир Привет", "hello🙂", "!\n"]:
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_boundaries_and_unknown_symbols(self):
        tokenizer = BPETokeniser().fit("abab", 100)
        tokens = tokenizer.encode("abab", add_bos=True, add_eos=True)
        self.assertEqual(tokens, [tokenizer.bos_id, 6, tokenizer.eos_id])
        self.assertEqual(tokenizer.decode(tokens), "abab")
        self.assertEqual(
            tokenizer.decode(tokens, skip_special_tokens=False), "<bos>abab<eos>"
        )
        self.assertEqual(tokenizer.encode("🙂"), [tokenizer.unk_id])
        self.assertEqual(tokenizer.decode([tokenizer.unk_id], False), "<unk>")
        self.assertEqual(tokenizer.decode([tokenizer.unk_id]), "")
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.encode("", True, True), [1, 2])

    def test_literal_special_token_names_remain_text(self):
        text = "<unk><bos><eos>"
        tokenizer = BPETokeniser().fit(text, 100)
        tokens = tokenizer.encode(text)
        self.assertTrue(all(token_id >= 3 for token_id in tokens))
        self.assertEqual(tokenizer.decode(tokens), text)

    def test_no_merges_across_text_boundaries(self):
        tokenizer = BPETokeniser().fit(["a", "b"], 100)
        self.assertEqual(tokenizer.merges, [])
        self.assertEqual(tokenizer.encode("ab"), [3, 4])

    def test_vocabulary_limit_includes_special_tokens(self):
        alphabet = [chr(0x400 + i) for i in range(98)]
        tokenizer = BPETokeniser().fit([alphabet[0] * 4] + alphabet[1:97], 100)
        self.assertEqual(tokenizer.vocab_size, 100)
        self.assertEqual(tokenizer.merges, [])
        tokenizer.fit([alphabet[0] * 4] + alphabet[1:96], 100)
        self.assertEqual(tokenizer.vocab_size, 100)
        self.assertEqual(len(tokenizer.merges), 1)
        with self.assertRaises(ValueError):
            tokenizer.fit(alphabet, 100)

    def test_invalid_fit_preserves_previous_state_and_refit_resets_it(self):
        tokenizer = BPETokeniser().fit("abab", 100)
        previous = (tokenizer.token_to_id.copy(), tokenizer.merges.copy())
        for text, size in [([], 100), ("", 100), (["a", 7], 100), (None, 100),
                           ("a", 99), ("a", 100.5), ("a", True)]:
            with self.subTest(text=text, size=size):
                with self.assertRaises(ValueError):
                    tokenizer.fit(text, size)
                self.assertEqual((tokenizer.token_to_id, tokenizer.merges), previous)
        tokenizer.fit("x", 100)
        self.assertEqual(tokenizer.encode("a"), [tokenizer.unk_id])
        self.assertEqual(tokenizer.encode("x"), [3])
        self.assertEqual(tokenizer.merges, [])

    def test_use_before_fit_and_invalid_ids(self):
        tokenizer = BPETokeniser()
        with self.assertRaises(RuntimeError):
            tokenizer.encode("a")
        with self.assertRaises(RuntimeError):
            tokenizer.decode([])
        tokenizer.fit("a", 100)
        for tokens in [[-1], [1000], [True], [1.5], "a"]:
            with self.subTest(tokens=tokens):
                with self.assertRaises(ValueError):
                    tokenizer.decode(tokens)
        with self.assertRaises(ValueError):
            tokenizer.encode(["a"])

    def test_merging_handles_overlap_and_preserves_list(self):
        tokenizer = BPETokeniser()
        for text, expected in [("", []), ("a", ["a"]),
                               ("aaa", ["aa", "a"]), ("aaaa", ["aa", "aa"])]:
            sequence = list(text)
            sequences = [sequence]
            tokenizer._merge_pairs(sequences, ("a", "a"), "aa")
            self.assertIs(sequences[0], sequence)
            self.assertEqual(sequence, expected)

    def test_save_and_load_preserve_behavior(self):
        tokenizer = BPETokeniser().fit(["Привет!", "abcabc", "<eos>"], 100)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = BPETokeniser.load(path)
        self.assertEqual(restored.token_to_id, tokenizer.token_to_id)
        self.assertEqual(restored.id_to_token, tokenizer.id_to_token)
        self.assertEqual(restored.merges, tokenizer.merges)
        for text in ["Привет!", "abc", "<eos>", "🙂", ""]:
            self.assertEqual(restored.encode(text, True, True), tokenizer.encode(text, True, True))
            self.assertEqual(restored.decode(restored.encode(text)), tokenizer.decode(tokenizer.encode(text)))

    def test_invalid_saved_rules_are_rejected(self):
        tokenizer = BPETokeniser().fit("abab", 100)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["merges"].reverse()
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                BPETokeniser.load(path)


if __name__ == "__main__":
    unittest.main()
