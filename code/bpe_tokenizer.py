"""
Byte Pair Encoding tokenizer for the HW2 NER scaffold.

Phase 1 intentionally favors correctness over speed. Phase 2 can replace the
naive merge loops with faster indexed structures while keeping this behavior as
the oracle.
"""

from collections import Counter, defaultdict
import heapq
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

import regex

from base_tokenizer import BaseTokenizer


SPACE_TOKEN = "▁"
BYTE_PREFIX = "\ue000"
PRETOKEN_PATTERN = r"\X"


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
        self.pretoken_pattern = PRETOKEN_PATTERN
        self._init_runtime_state()

    def __getstate__(self) -> Dict:
        """Serialize primitive state; drop runtime cache and compiled regex."""
        return {
            "vocab_size": self.vocab_size,
            "space_token": self.space_token,
            "token_to_id": dict(self.token_to_id),
            "id_to_token": dict(self.id_to_token),
            "special_tokens": dict(self.special_tokens),
            "merges": list(self.merges),
            "merge_ranks": dict(self.merge_ranks),
            "byte_tokens": dict(self.byte_tokens),
            "pretoken_pattern": self.pretoken_pattern,
        }

    def __setstate__(self, state: Dict) -> None:
        """Restore primitive state and rebuild runtime helpers."""
        self.vocab_size = state["vocab_size"]
        self.space_token = state["space_token"]
        self.token_to_id = dict(state["token_to_id"])
        self.id_to_token = dict(state["id_to_token"])
        self.special_tokens = dict(state["special_tokens"])
        self.merges = list(state["merges"])
        self.merge_ranks = dict(state["merge_ranks"])
        self.byte_tokens = dict(state["byte_tokens"])
        self.pretoken_pattern = state.get("pretoken_pattern", PRETOKEN_PATTERN)
        self._init_runtime_state()

    def train(self, texts: List[str]) -> None:
        """Train a deterministic char-up BPE tokenizer on weighted unique texts."""
        self._reset_vocab()
        sequences, weights = self._weighted_sequences(texts)

        base_tokens = set(self.byte_tokens.values())
        base_tokens.add(self.space_token)
        for sequence in sequences:
            base_tokens.update(sequence)
        for token in sorted(base_tokens):
            self._add_token(token)

        pair_counts, pair_to_sequences, pair_heap = self._build_pair_index(sequences, weights)
        while len(self.token_to_id) < self.vocab_size:
            if len(self.token_to_id) == self.vocab_size - 1 and not self._has_bigram_token():
                self._force_bigram(sequences)
                break
            pair = self._best_indexed_pair(pair_counts, pair_heap)
            if pair is None:
                break
            self._record_merge(pair)
            self._apply_indexed_merge(sequences, weights, pair, pair_counts, pair_to_sequences, pair_heap)

        if len(self.token_to_id) < self.vocab_size and not self._has_bigram_token():
            self._force_bigram(sequences)

    def encode(self, text: str) -> List[int]:
        """Convert a text string into token IDs, never emitting [UNK]."""
        cached = self._encode_cache.get(text)
        if cached is not None:
            return list(cached)
        ids = self._encode_uncached(text)
        self._encode_cache[text] = list(ids)
        return list(ids)

    def _encode_uncached(self, text: str) -> List[int]:
        """Encode without consulting or populating the runtime cache."""
        symbols = self._text_to_encode_symbols(text)
        symbols = self._apply_ranked_merges(symbols)
        return self._ids_from_symbols(symbols)

    def _ids_from_symbols(self, symbols: Iterable[str]) -> List[int]:
        """Map final symbols to token IDs with UTF-8 byte fallback."""
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
        self.pretoken_pattern = PRETOKEN_PATTERN
        self._init_runtime_state()

    def _init_runtime_state(self) -> None:
        self._pretoken_regex = regex.compile(self.pretoken_pattern)
        self._encode_cache: Dict[str, List[int]] = {}

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

    def _weighted_sequences(self, texts: Sequence[str]) -> Tuple[List[List[str]], List[int]]:
        text_counts = Counter(text for text in texts if text)
        sequences: List[List[str]] = []
        weights: List[int] = []
        for text in sorted(text_counts):
            sequences.append(self._text_to_train_symbols(text))
            weights.append(text_counts[text])
        return sequences, weights

    def _build_pair_index(
        self, sequences: Sequence[Sequence[str]], weights: Sequence[int]
    ) -> Tuple[Counter, DefaultDict[Tuple[str, str], Set[int]], List[Tuple[int, str, str, Tuple[str, str]]]]:
        pair_counts: Counter = Counter()
        pair_to_sequences: DefaultDict[Tuple[str, str], Set[int]] = defaultdict(set)
        pair_heap: List[Tuple[int, str, str, Tuple[str, str]]] = []
        for sequence_id, sequence in enumerate(sequences):
            self._add_sequence_pairs(
                sequence_id,
                sequence,
                weights[sequence_id],
                pair_counts,
                pair_to_sequences,
                pair_heap,
            )
        return pair_counts, pair_to_sequences, pair_heap

    def _sequence_pair_counts(self, sequence: Sequence[str]) -> Counter:
        counts: Counter = Counter()
        for index in range(len(sequence) - 1):
            pair = (sequence[index], sequence[index + 1])
            if self._legal_merge(pair[0], pair[1]):
                counts[pair] += 1
        return counts

    def _push_pair_heap(
        self,
        pair: Tuple[str, str],
        pair_counts: Counter,
        pair_heap: List[Tuple[int, str, str, Tuple[str, str]]],
    ) -> None:
        count = pair_counts.get(pair, 0)
        if count > 0:
            heapq.heappush(pair_heap, (-count, pair[0], pair[1], pair))

    def _add_sequence_pairs(
        self,
        sequence_id: int,
        sequence: Sequence[str],
        weight: int,
        pair_counts: Counter,
        pair_to_sequences: DefaultDict[Tuple[str, str], Set[int]],
        pair_heap: List[Tuple[int, str, str, Tuple[str, str]]],
    ) -> None:
        for pair, count in self._sequence_pair_counts(sequence).items():
            pair_counts[pair] += count * weight
            pair_to_sequences[pair].add(sequence_id)
            self._push_pair_heap(pair, pair_counts, pair_heap)

    def _remove_sequence_pairs(
        self,
        sequence_id: int,
        sequence: Sequence[str],
        weight: int,
        pair_counts: Counter,
        pair_to_sequences: DefaultDict[Tuple[str, str], Set[int]],
        pair_heap: List[Tuple[int, str, str, Tuple[str, str]]],
    ) -> None:
        for pair, count in self._sequence_pair_counts(sequence).items():
            pair_counts[pair] -= count * weight
            if pair_counts[pair] <= 0:
                pair_counts.pop(pair, None)
            members = pair_to_sequences.get(pair)
            if members is not None:
                members.discard(sequence_id)
                if not members:
                    pair_to_sequences.pop(pair, None)
            self._push_pair_heap(pair, pair_counts, pair_heap)

    def _best_indexed_pair(
        self, pair_counts: Counter, pair_heap: List[Tuple[int, str, str, Tuple[str, str]]]
    ) -> Tuple[str, str] | None:
        while pair_heap:
            negative_count, _left, _right, pair = heapq.heappop(pair_heap)
            count = pair_counts.get(pair, 0)
            if count <= 0 or -negative_count != count:
                continue
            if pair in self.merge_ranks or not self._legal_merge(pair[0], pair[1]):
                continue
            return pair
        return None

    def _apply_indexed_merge(
        self,
        sequences: List[List[str]],
        weights: Sequence[int],
        pair: Tuple[str, str],
        pair_counts: Counter,
        pair_to_sequences: DefaultDict[Tuple[str, str], Set[int]],
        pair_heap: List[Tuple[int, str, str, Tuple[str, str]]],
    ) -> None:
        impacted = sorted(pair_to_sequences.get(pair, set()))
        for sequence_id in impacted:
            sequence = sequences[sequence_id]
            self._remove_sequence_pairs(
                sequence_id,
                sequence,
                weights[sequence_id],
                pair_counts,
                pair_to_sequences,
                pair_heap,
            )
            merged_sequence = self._apply_pair(sequence, pair)
            sequences[sequence_id] = merged_sequence
            self._add_sequence_pairs(
                sequence_id,
                merged_sequence,
                weights[sequence_id],
                pair_counts,
                pair_to_sequences,
                pair_heap,
            )

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
