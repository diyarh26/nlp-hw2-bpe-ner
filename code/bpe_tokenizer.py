"""
Byte Pair Encoding tokenizer for the HW2 NER scaffold.

Training is a deterministic char-up BPE over weighted word chunks. The merge
loop uses an integer-symbol representation with local-neighbor pair-count deltas
and a lazy max-heap, so it scales with the number of UNIQUE word chunks rather
than the raw corpus. It produces merges/vocab byte-identical to the slow
full-recount oracle in speed_check.py (the behavioral ground truth).
"""

from collections import Counter
import heapq
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import regex

from base_tokenizer import BaseTokenizer


SPACE_TOKEN = "▁"
BYTE_PREFIX = ""
# Batch 2 — structured (GPT-2-style, Unicode-aware) pre-tokenization. Used ONLY at
# TRAIN time (in `_text_to_chunks`) to bound which adjacent pairs may be counted/
# merged: letters, digits, and punctuation/symbol runs form separate chunks, and a
# leading space binds to the following piece (mapped to the ▁ marker). Splitting at
# letter/digit/punct boundaries keeps merges inside coherent units so word-initial
# subtokens stay consistent for the NER first-subtoken labeller. encode/decode are
# unchanged: encode still applies the learned merges greedily over the flat symbol
# sequence, so only merges training actually learned (within-category, plus the one
# forced cross-word bigram) can fire — keeping the slow/fast encode oracle and
# length-faithful decode green by construction. Stored as a string (pickle-safe),
# compiled in `_init_runtime_state`.
PRETOKEN_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)|"  # common English contractions
    r" ?\p{L}+|"               # optional leading space + a run of letters
    r" ?\p{N}+|"               # optional leading space + a run of digits
    r" ?[^\s\p{L}\p{N}]+|"     # optional leading space + punctuation/symbols (@,#,emoji,...)
    r"\s+(?!\S)|"              # trailing whitespace run
    r"\s+"                      # any remaining whitespace run
)
FORCED_BIGRAM_SAMPLE_LINES = 4000


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
        """Train a deterministic char-up BPE tokenizer on weighted word chunks."""
        self._reset_vocab()
        sequences, weights = self._weighted_chunk_sequences(texts)

        base_tokens = set(self.byte_tokens.values())
        base_tokens.add(self.space_token)
        for sequence in sequences:
            base_tokens.update(sequence)
        for token in sorted(base_tokens):
            self._add_token(token)

        self._train_merges(sequences, weights, texts)

    def _train_merges(
        self,
        sequences: Sequence[Sequence[str]],
        weights: Sequence[int],
        texts: Sequence[str],
    ) -> None:
        """Efficient incremental BPE merge loop (identical result to the oracle).

        Symbols are interned to dense integer ids. After each merge only the
        local neighbour pairs at each merge site are updated (no whole-chunk
        recount), and only pairs whose count changed are re-pushed onto a lazy
        max-heap. ``pair_chunks`` membership is kept as a superset (entries are
        only added, never pruned on decrement), so a merge never misses a chunk
        that still contains the pair; stale entries are harmless no-op scans.
        """
        space_token = self.space_token

        str_to_local: Dict[str, int] = {}
        local_to_str: List[str] = []
        local_spaces: List[int] = []

        def intern(symbol: str) -> int:
            local_id = str_to_local.get(symbol)
            if local_id is None:
                local_id = len(local_to_str)
                str_to_local[symbol] = local_id
                local_to_str.append(symbol)
                local_spaces.append(symbol.count(space_token))
            return local_id

        seqs: List[List[int]] = [[intern(sym) for sym in seq] for seq in sequences]

        pair_counts: Dict[Tuple[int, int], int] = {}
        pair_chunks: Dict[Tuple[int, int], Set[int]] = {}
        heap: List[Tuple[int, str, str, Tuple[int, int]]] = []

        for chunk_id, seq in enumerate(seqs):
            weight = weights[chunk_id]
            for index in range(len(seq) - 1):
                left = seq[index]
                right = seq[index + 1]
                if local_spaces[left] + local_spaces[right] > 1:
                    continue
                key = (left, right)
                pair_counts[key] = pair_counts.get(key, 0) + weight
                members = pair_chunks.get(key)
                if members is None:
                    members = set()
                    pair_chunks[key] = members
                members.add(chunk_id)
        for key, count in pair_counts.items():
            heapq.heappush(heap, (-count, local_to_str[key[0]], local_to_str[key[1]], key))

        changed: Set[Tuple[int, int]] = set()

        def adjust(left: int, right: int, chunk_id: int, delta: int) -> None:
            if local_spaces[left] + local_spaces[right] > 1:
                return
            key = (left, right)
            new_count = pair_counts.get(key, 0) + delta
            if new_count <= 0:
                pair_counts.pop(key, None)
                pair_chunks.pop(key, None)
            else:
                pair_counts[key] = new_count
                if delta > 0:
                    members = pair_chunks.get(key)
                    if members is None:
                        members = set()
                        pair_chunks[key] = members
                    members.add(chunk_id)
            changed.add(key)

        target = self.vocab_size
        while len(self.token_to_id) < target:
            if len(self.token_to_id) == target - 1 and not self._has_bigram_token():
                pair = self._forced_cross_word_bigram_pair(texts)
                if pair is not None:
                    self._record_merge(pair)
                break

            best_key = None
            while heap:
                neg_count, _left_str, _right_str, key = heapq.heappop(heap)
                live = pair_counts.get(key, 0)
                if live <= 0 or -neg_count != live:
                    continue
                best_key = key
                break
            if best_key is None:
                break

            left_id, right_id = best_key
            left_str = local_to_str[left_id]
            right_str = local_to_str[right_id]
            self._record_merge((left_str, right_str))
            merged_id = intern(left_str + right_str)

            chunk_ids = pair_chunks.pop(best_key, None)
            pair_counts.pop(best_key, None)
            if not chunk_ids:
                continue

            changed.clear()
            for chunk_id in chunk_ids:
                seq = seqs[chunk_id]
                length = len(seq)
                weight = weights[chunk_id]
                new_seq: List[int] = []
                index = 0
                while index < length:
                    if (
                        index + 1 < length
                        and seq[index] == left_id
                        and seq[index + 1] == right_id
                    ):
                        left_neighbor = new_seq[-1] if new_seq else None
                        right_neighbor = seq[index + 2] if index + 2 < length else None
                        if left_neighbor is not None:
                            adjust(left_neighbor, left_id, chunk_id, -weight)
                        if right_neighbor is not None:
                            adjust(right_id, right_neighbor, chunk_id, -weight)
                        new_seq.append(merged_id)
                        if left_neighbor is not None:
                            adjust(left_neighbor, merged_id, chunk_id, weight)
                        if right_neighbor is not None:
                            adjust(merged_id, right_neighbor, chunk_id, weight)
                        index += 2
                    else:
                        new_seq.append(seq[index])
                        index += 1
                seqs[chunk_id] = new_seq

            for key in changed:
                count = pair_counts.get(key, 0)
                if count > 0:
                    heapq.heappush(heap, (-count, local_to_str[key[0]], local_to_str[key[1]], key))

        if len(self.token_to_id) < target and not self._has_bigram_token():
            pair = self._forced_cross_word_bigram_pair(texts)
            if pair is not None:
                self._record_merge(pair)

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

    def _text_to_chunks(self, text: str) -> List[List[str]]:
        """Split text into pre-token chunks via the structured regex, each chunk a
        list of char symbols (a literal space -> the ▁ marker). The regex tiles the
        whole string, so concatenating the chunks reconstructs the text (no symbol
        dropped); it only bounds which adjacent pairs training may count/merge."""
        chunks: List[List[str]] = []
        for match in self._pretoken_regex.finditer(text):
            piece = match.group()
            if not piece:
                continue
            chunks.append([self.space_token if char == " " else char for char in piece])
        return chunks

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

    def _weighted_chunk_sequences(self, texts: Sequence[str]) -> Tuple[List[List[str]], List[int]]:
        chunk_counts: Counter = Counter()
        for text in texts:
            if not text:
                continue
            for chunk in self._text_to_chunks(text):
                chunk_counts[tuple(chunk)] += 1
        sequences: List[List[str]] = []
        weights: List[int] = []
        for chunk in sorted(chunk_counts):
            sequences.append(list(chunk))
            weights.append(chunk_counts[chunk])
        return sequences, weights

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

    def _forced_cross_word_bigram_pair(self, texts: Sequence[str]) -> Tuple[str, str] | None:
        line_counts = Counter(text for text in texts if text)
        ordered = sorted(line_counts.items(), key=lambda item: (-item[1], item[0]))
        sample = [line for line, _count in ordered[:FORCED_BIGRAM_SAMPLE_LINES]]
        counts: Counter = Counter()
        for line in sample:
            symbols = self._apply_ranked_merges(self._text_to_encode_symbols(line))
            for index in range(len(symbols) - 1):
                left = symbols[index]
                right = symbols[index + 1]
                pair = (left, right)
                if pair in self.merge_ranks or not self._legal_merge(left, right):
                    continue
                surface = (left + right).replace(self.space_token, " ").strip()
                if surface.count(" ") == 1:
                    counts[pair] += 1
        if not counts:
            return None
        return min(counts.items(), key=_pair_sort_key)[0]

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
