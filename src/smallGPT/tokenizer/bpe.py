import json
from collections import defaultdict
from pathlib import Path


class BPETokeniser:
    """Учебный посимвольный BPE без PyTorch и готовых tokenizer-библиотек."""

    SPECIAL_TOKENS = ("<unk>", "<bos>", "<eos>")

    def __init__(self):
        self.unk_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []
        self._is_fitted = False

    @property
    def vocab_size(self):
        """Фактическое число токенов, включая UNK, BOS и EOS."""
        return len(self.token_to_id)

    def fit(self, text, vocab_size):
        """Обучить BPE на строке или списке историй; повторный вызов обучает заново.

        vocab_size — максимальный размер словаря. Если пары закончились,
        итоговый словарь может быть меньше. Исходные тексты не изменяются.
        """
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 100:
            raise ValueError("Vocab size must be an integer greater than or equal to 100")
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) or not all(isinstance(sequence, str) for sequence in text):
            raise ValueError("Text must be a string or list of strings")
        if not text or not any(text):
            raise ValueError("Text must contain at least one character")

        sequences = [list(sequence) for sequence in text]
        token_to_id = {
            token: token_id for token_id, token in enumerate(self.SPECIAL_TOKENS)
        }
        id_to_token = {
            token_id: token for token, token_id in token_to_id.items()
        }

        for sequence in sequences:
            for char in sequence:
                if char not in token_to_id:
                    token_id = len(token_to_id)
                    token_to_id[char] = token_id
                    id_to_token[token_id] = char
        if len(token_to_id) > vocab_size:
            raise ValueError("Vocab size must fit all unique characters and special tokens")

        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.merges = []

        while len(self.token_to_id) < vocab_size:
            # Буквальный текст "<eos>" должен оставаться обычным текстом,
            # а не превращаться в управляющий токен с eos_id.
            pairs = {
                pair: count
                for pair, count in self._count_pairs(sequences).items()
                if pair[0] + pair[1] not in self.SPECIAL_TOKENS
            }
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[new_token] = token_id
                self.id_to_token[token_id] = new_token
            self._merge_pairs(sequences, best_pair, new_token)
            self.merges.append(best_pair)
        self._is_fitted = True
        return self

    def encode(self, text, add_bos=False, add_eos=False):
        """Вернуть список ID, применяя правила в порядке обучения.

        Неизвестные символы заменяются UNK, поэтому их восстановить нельзя.
        Для целой истории можно включить BOS и EOS; для prompt EOS не нужен.
        """
        self._require_fitted()
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        if not isinstance(add_bos, bool) or not isinstance(add_eos, bool):
            raise ValueError("add_bos and add_eos must be booleans")

        sequence = list(text)
        for pair in self.merges:
            self._merge_pairs([sequence], pair, pair[0] + pair[1])

        tokens = [self.token_to_id.get(token, self.unk_id) for token in sequence]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens, skip_special_tokens=True):
        """Восстановить строку из ID; по умолчанию пропустить UNK, BOS и EOS.

        skip_special_tokens=False отображает их как <unk>, <bos> и <eos>.
        Неизвестный ID — ошибка: его нет в словаре этого tokenizer.
        """
        self._require_fitted()
        if not isinstance(tokens, (list, tuple)):
            raise ValueError("Tokens must be a list or tuple of integer IDs")
        if not isinstance(skip_special_tokens, bool):
            raise ValueError("skip_special_tokens must be a boolean")

        parts = []
        for token_id in tokens:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError("Token IDs must be integers")
            if token_id not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {token_id}")
            if skip_special_tokens and token_id in (self.unk_id, self.bos_id, self.eos_id):
                continue
            parts.append(self.id_to_token[token_id])
        return "".join(parts)

    def save(self, path):
        """Сохранить словарь и порядок объединений в JSON."""
        self._require_fitted()
        data = {
            "version": 1,
            "tokens": [self.id_to_token[i] for i in range(self.vocab_size)],
            "merges": self.merges,
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        """Загрузить обученный tokenizer; ID и порядок правил сохраняются."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported tokenizer format")
        tokens = data.get("tokens")
        merges = data.get("merges")
        if (
            not isinstance(tokens, list)
            or len(tokens) <= len(cls.SPECIAL_TOKENS)
            or not all(isinstance(token, str) and token for token in tokens)
            or tokens[:3] != list(cls.SPECIAL_TOKENS)
            or len(set(tokens)) != len(tokens)
            or not isinstance(merges, list)
        ):
            raise ValueError("Invalid tokenizer vocabulary")

        # Каждое правило использует символы или уже выученные токены.
        available = {token for token in tokens if len(token) == 1}
        rules = []
        for pair in merges:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(token, str) and token in available for token in pair)
            ):
                raise ValueError("Invalid merge rule")
            new_token = pair[0] + pair[1]
            if new_token in cls.SPECIAL_TOKENS:
                raise ValueError("Merge rules must not produce special tokens")
            available.add(new_token)
            rules.append(tuple(pair))
        if available | set(cls.SPECIAL_TOKENS) != set(tokens):
            raise ValueError("Vocabulary does not match merge rules")

        tokenizer = cls()
        tokenizer.token_to_id = {token: i for i, token in enumerate(tokens)}
        tokenizer.id_to_token = dict(enumerate(tokens))
        tokenizer.merges = rules
        tokenizer._is_fitted = True
        return tokenizer

    def _require_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit before using the tokenizer")

    def _count_pairs(self, sequences):
        """Посчитать соседние пары, не пересекая границы текстов."""
        pairs = defaultdict(int)
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                pairs[pair] += 1
        return pairs

    def _merge_pairs(self, sequences, best_pair, new_token):
        """Объединить непересекающиеся пары слева направо, изменив списки.

        Результат строится за один проход; исходный список сохраняет свой ID.
        """
        for sequence in sequences:
            merged = []
            i = 0
            while i < len(sequence):
                if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == best_pair:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(sequence[i])
                    i += 1
            sequence[:] = merged
