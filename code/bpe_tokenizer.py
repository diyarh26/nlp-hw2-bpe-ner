"""
Byte Pair Encoding tokenizer for the HW2 NER scaffold.

Phase 1 intentionally favors correctness over speed. Phase 2 can replace the
naive merge loops with faster indexed structures while keeping this behavior as
the oracle.
"""

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from base_tokenizer import BaseTokenizer


SPACE_TOKEN = "▁"
BYTE_PREFIX = "\ue000"
TRAIN_TEXT_LIMIT = 5000


def _pair_sort_key(item: Tuple[Tuple[str, str], int]) -> Tuple[int, str, str]:
    pair, count = item
    return (-count, pair[0], pair[1])


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.space_token = SPACE_TOKEN
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.byte_tokens = {i: f"{BYTE_PREFIX}{i:02x}" for i in range(256)}

    def train(self, texts: List[str]) -> None:
        """Train a deterministic char-up BPE tokenizer on a capped text slice."""
        self._reset_vocab()
        limited_texts = list(texts[:TRAIN_TEXT_LIMIT])
        sequences = [self._text_to_train_symbols(text) for text in limited_texts if text]

        base_tokens = set(self.byte_tokens.values())
        base_tokens.add(self.space_token)
        for sequence in sequences:
            base_tokens.update(sequence)
        for token in sorted(base_tokens):
            self._add_token(token)

        while len(self.token_to_id) < self.vocab_size:
            pair_counts = self._count_pairs(sequences)
            candidates = [
                (pair, count)
                for pair, count in pair_counts.items()
                if count > 0
                and pair not in self.merge_ranks
                and self._legal_merge(pair[0], pair[1])
            ]
            if not candidates:
                break
            pair, _count = min(candidates, key=_pair_sort_key)
            self._record_merge(pair)
            sequences = self._apply_pair_to_sequences(sequences, pair)

        if not self._has_bigram_token():
            self._force_bigram(sequences)

    def encode(self, text: str) -> List[int]:
        """Convert a text string into token IDs, never emitting [UNK]."""
        symbols = self._text_to_encode_symbols(text)
        symbols = self._apply_ranked_merges(symbols)
        ids: List[int] = []
        for symbol in symbols:
            token_id = self.token_to_id.get(symbol)
            if token_id is None:
                ids.extend(self.token_to_id[self.byte_tokens[byte]] for byte in symbol.encode("utf-8"))
            else:
                ids.append(token_id)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text with exact byte-fallback recovery."""
        pieces: List[str] = []
        for token_id in token_ids:
            if token_id in self.special_tokens.values():
                continue
            token = self.id_to_token.get(token_id)
            if token is not None:
                pieces.append(token)
        return self._decode_surface("".join(pieces))

    def _reset_vocab(self) -> None:
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {token_id: token for token, token_id in self.special_tokens.items()}
        self.merges = []
        self.merge_ranks = {}
        self.space_token = SPACE_TOKEN
        self.byte_tokens = {i: f"{BYTE_PREFIX}{i:02x}" for i in range(256)}

    def _add_token(self, token: str) -> None:
        if token in self.token_to_id:
            return
        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token

    def _record_merge(self, pair: Tuple[str, str]) -> str:
        merged = pair[0] + pair[1]
        self.merge_ranks[pair] = len(self.merges)
        self.merges.append(pair)
        self._add_token(merged)
        return merged

    def _text_to_train_symbols(self, text: str) -> List[str]:
        return [self.space_token if char == " " else char for char in text]

    def _text_to_encode_symbols(self, text: str) -> List[str]:
        symbols: List[str] = []
        for char in text:
            if char == " ":
                symbols.append(self.space_token)
            elif char in self.token_to_id:
                symbols.append(char)
            else:
                symbols.extend(self.byte_tokens[byte] for byte in char.encode("utf-8"))
        return symbols

    def _count_pairs(self, sequences: Sequence[Sequence[str]]) -> Counter:
        counts: Counter = Counter()
        for sequence in sequences:
            for index in range(len(sequence) - 1):
                pair = (sequence[index], sequence[index + 1])
                if self._legal_merge(pair[0], pair[1]):
                    counts[pair] += 1
        return counts

    def _legal_merge(self, left: str, right: str) -> bool:
        return (left + right).count(self.space_token) <= 1

    def _apply_pair_to_sequences(
        self, sequences: Sequence[Sequence[str]], pair: Tuple[str, str]
    ) -> List[List[str]]:
        return [self._apply_pair(sequence, pair) for sequence in sequences]

    def _apply_pair(self, sequence: Sequence[str], pair: Tuple[str, str]) -> List[str]:
        merged = pair[0] + pair[1]
        output: List[str] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                output.append(merged)
                index += 2
            else:
                output.append(sequence[index])
                index += 1
        return output

    def _apply_ranked_merges(self, symbols: List[str]) -> List[str]:
        while True:
            best_rank = None
            best_index = None
            best_pair = None
            for index in range(len(symbols) - 1):
                pair = (symbols[index], symbols[index + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None or not self._legal_merge(pair[0], pair[1]):
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = index
                    best_pair = pair
            if best_pair is None or best_index is None:
                break
            symbols = (
                symbols[:best_index]
                + [best_pair[0] + best_pair[1]]
                + symbols[best_index + 2 :]
            )
        return symbols

    def _has_bigram_token(self) -> bool:
        for token in self._non_special_tokens():
            surface = token.replace(self.space_token, " ").strip()
            if surface.count(" ") == 1:
                return True
        return False

    def _force_bigram(self, sequences: Sequence[Sequence[str]]) -> None:
        pair_counts = self._count_pairs(sequences)
        candidates = [
            (pair, count)
            for pair, count in pair_counts.items()
            if count > 0
            and pair not in self.merge_ranks
            and self._legal_merge(pair[0], pair[1])
            and self.space_token in (pair[0] + pair[1])
        ]
        if not candidates:
            return
        pair, _count = min(candidates, key=_pair_sort_key)
        self._record_merge(pair)

    def _non_special_tokens(self) -> Iterable[str]:
        for token in self.token_to_id:
            if token not in self.special_tokens:
                yield token

    def _decode_surface(self, surface: str) -> str:
        output: List[str] = []
        pending_bytes = bytearray()
        index = 0

        def flush_bytes() -> None:
            if pending_bytes:
                output.append(bytes(pending_bytes).decode("utf-8", errors="replace"))
                pending_bytes.clear()

        while index < len(surface):
            if surface.startswith(BYTE_PREFIX, index) and index + 3 <= len(surface):
                hex_part = surface[index + len(BYTE_PREFIX) : index + len(BYTE_PREFIX) + 2]
                if len(hex_part) == 2 and all(char in "0123456789abcdef" for char in hex_part):
                    pending_bytes.append(int(hex_part, 16))
                    index += len(BYTE_PREFIX) + 2
                    continue
            flush_bytes()
            char = surface[index]
            output.append(" " if char == self.space_token else char)
            index += 1
        flush_bytes()
        return "".join(output)
