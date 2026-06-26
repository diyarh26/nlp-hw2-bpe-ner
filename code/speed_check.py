"""Phase 2 encode oracle and speed/cache probes.

This harness is intentionally framework-free, like align_check.py. It gives the
optimizer work a behavioral oracle before code/speed changes land.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from bpe_tokenizer import BPETokenizer  # noqa: E402


Check = Callable[[], None]


def read_lines(path: Path, limit: int = 2000) -> List[str]:
    lines: List[str] = []
    if not path.exists():
        return lines
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if text:
                lines.append(text)
            if len(lines) >= limit:
                break
    return lines


def read_tagged_texts(path: Path, limit: int = 200) -> List[str]:
    texts: List[str] = []
    words: List[str] = []
    if not path.exists():
        return texts
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                if words:
                    texts.append(" ".join(words))
                    words = []
                if len(texts) >= limit:
                    break
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                words.append(parts[0])
    if words and len(texts) < limit:
        texts.append(" ".join(words))
    return texts


def has_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def collect_samples() -> List[str]:
    raw_lines = (
        read_lines(ROOT_DIR / "data" / "domain_1_dev.txt")
        + read_lines(ROOT_DIR / "data" / "domain_2_dev.txt")
    )
    tagged_texts = (
        read_tagged_texts(ROOT_DIR / "data" / "ner_data" / "dev_1_binary.tagged")
        + read_tagged_texts(ROOT_DIR / "data" / "ner_data" / "dev_2_binary.tagged")
    )

    samples: List[str] = []
    samples.extend(raw_lines[:40])
    samples.extend(tagged_texts[:40])

    non_ascii = next((line for line in raw_lines + tagged_texts if has_non_ascii(line)), None)
    double_space = next((line for line in raw_lines if "  " in line), None)
    if non_ascii is None:
        raise AssertionError("sample fixture missing non-ASCII line from data/")
    if double_space is None:
        raise AssertionError("sample fixture missing double-space line from raw data/")

    samples.extend(
        [
            non_ascii,
            double_space,
            "OOV stress: caf\u00e9 na\u00efve \U0001f600 \u0639\u0631\u0628\u064a",
            "repeat repeat repeat in the world",
        ]
    )
    return list(dict.fromkeys(samples))


def build_probe_tokenizer(samples: List[str] | None = None) -> BPETokenizer:
    if samples is None:
        samples = collect_samples()
    tokenizer = BPETokenizer(vocab_size=800)
    training_lines = [
        "in the house",
        "in the car",
        "in the world",
        "the end",
        "hello world",
        "repeat repeat repeat",
    ] * 40
    training_lines.extend(samples[:80])
    tokenizer.train(training_lines)
    return tokenizer


def probe_training_lines(samples: List[str]) -> List[str]:
    training_lines = [
        "in the house",
        "in the car",
        "in the world",
        "the end",
        "hello world",
        "repeat repeat repeat",
    ] * 40
    training_lines.extend(samples[:80])
    return training_lines


def ids_from_symbols(tokenizer: BPETokenizer, symbols: Iterable[str]) -> List[int]:
    ids: List[int] = []
    for symbol in symbols:
        token_id = tokenizer.token_to_id.get(symbol)
        if token_id is None:
            ids.extend(tokenizer.token_to_id[tokenizer.byte_tokens[byte]] for byte in symbol.encode("utf-8"))
        else:
            ids.append(token_id)
    return ids


def slow_replay_encode(tokenizer: BPETokenizer, text: str) -> List[int]:
    symbols = tokenizer._text_to_encode_symbols(text)
    for pair in tokenizer.merges:
        symbols = tokenizer._apply_pair(symbols, pair)
    return ids_from_symbols(tokenizer, symbols)


def train_full_recount_oracle(tokenizer: BPETokenizer, texts: List[str]) -> None:
    tokenizer._reset_vocab()
    sequences = [
        chunk
        for text in texts
        if text
        for chunk in tokenizer._text_to_chunks(text)
    ]

    base_tokens = set(tokenizer.byte_tokens.values())
    base_tokens.add(tokenizer.space_token)
    for sequence in sequences:
        base_tokens.update(sequence)
    for token in sorted(base_tokens):
        tokenizer._add_token(token)

    while len(tokenizer.token_to_id) < tokenizer.vocab_size:
        if len(tokenizer.token_to_id) == tokenizer.vocab_size - 1 and not tokenizer._has_bigram_token():
            pair = tokenizer._forced_cross_word_bigram_pair(texts)
            if pair is not None:
                tokenizer._record_merge(pair)
            break
        pair_counts = tokenizer._count_pairs(sequences)
        candidates = [
            (pair, count)
            for pair, count in pair_counts.items()
            if count > 0
            and pair not in tokenizer.merge_ranks
            and tokenizer._legal_merge(pair[0], pair[1])
        ]
        if not candidates:
            break
        pair, _count = min(candidates, key=lambda item: (-item[1], item[0][0], item[0][1]))
        tokenizer._record_merge(pair)
        sequences = tokenizer._apply_pair_to_sequences(sequences, pair)

    if len(tokenizer.token_to_id) < tokenizer.vocab_size and not tokenizer._has_bigram_token():
        pair = tokenizer._forced_cross_word_bigram_pair(texts)
        if pair is not None:
            tokenizer._record_merge(pair)


def check_oracle() -> None:
    samples = collect_samples()
    tokenizer = build_probe_tokenizer(samples)
    for text in samples:
        public_ids = tokenizer.encode(text)
        oracle_ids = slow_replay_encode(tokenizer, text)
        if public_ids != oracle_ids:
            raise AssertionError(f"oracle mismatch for {text!r}")
        reconstructed = tokenizer.decode(public_ids)
        if reconstructed.replace(" ", "") != text.replace(" ", ""):
            raise AssertionError(f"decode reconstruction mismatch for {text!r}")
    print(f"ORACLE: PASS - {len(samples)} samples matched slow replay")


def uncached_encode(tokenizer: BPETokenizer, text: str) -> List[int]:
    helper = getattr(tokenizer, "_encode_uncached", None)
    if helper is not None:
        return list(helper(text))
    return slow_replay_encode(tokenizer, text)


def check_cache() -> None:
    samples = collect_samples()
    tokenizer = build_probe_tokenizer(samples)
    repeated = (samples[:12] + ["repeat repeat repeat in the world"]) * 200

    tokenizer._encode_cache.clear()
    start = time.perf_counter()
    uncached_outputs = [uncached_encode(tokenizer, text) for text in repeated]
    uncached_seconds = time.perf_counter() - start

    tokenizer._encode_cache.clear()
    start = time.perf_counter()
    public_outputs = [tokenizer.encode(text) for text in repeated]
    cached_seconds = time.perf_counter() - start
    cache_entries = len(getattr(tokenizer, "_encode_cache", {}))

    if public_outputs != uncached_outputs:
        raise AssertionError("public encode output differs from uncached/oracle output")
    if cached_seconds >= uncached_seconds:
        raise AssertionError(
            f"cached encode was not faster: cached={cached_seconds:.6f}, "
            f"uncached={uncached_seconds:.6f}"
        )

    tokenizer._encode_cache["sentinel"] = [1]
    loaded = pickle.loads(pickle.dumps(tokenizer))
    if getattr(loaded, "_encode_cache", None):
        raise AssertionError("_encode_cache persisted across pickle round-trip")

    print(f"uncached_seconds={uncached_seconds:.6f}")
    print(f"cached_seconds={cached_seconds:.6f}")
    print(f"cache_entries={cache_entries}")
    print("CACHE: PASS - cached encode faster, output equality, pickle cache drop verified")


def check_train() -> None:
    samples = collect_samples()
    training_lines = [
        "in the house",
        "in the car",
        "in the world",
        "the end",
        "hello world",
        "repeat repeat repeat",
    ] * 20
    training_lines.extend(samples[:10])
    incremental = BPETokenizer(vocab_size=420)
    full_recount = BPETokenizer(vocab_size=420)
    incremental.train(training_lines)
    train_full_recount_oracle(full_recount, training_lines)

    if incremental.merges != full_recount.merges:
        raise AssertionError("incremental merges differ from full-recount oracle")
    if incremental.token_to_id != full_recount.token_to_id:
        raise AssertionError("incremental vocab differs from full-recount oracle")
    for text in samples[:25]:
        if incremental.encode(text) != full_recount.encode(text):
            raise AssertionError(f"incremental encode differs from oracle for {text!r}")
    if not incremental.merges:
        raise AssertionError("probe tokenizer learned no merges")
    if not incremental._has_bigram_token():
        raise AssertionError("probe tokenizer learned no legal bigram")
    print(f"TRAIN: PASS - incremental trainer matched full recount for {len(incremental.merges)} merges")


CHECKS: dict[str, Check] = {
    "oracle": check_oracle,
    "cache": check_cache,
    "train": check_train,
}


def parse_selected(raw: str) -> List[str]:
    if raw == "all":
        return list(CHECKS)
    selected = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(CHECKS))
    if unknown:
        raise SystemExit(f"Unknown check(s): {', '.join(unknown)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 tokenizer speed/oracle checks.")
    parser.add_argument("--only", default="all", help="Comma-separated: oracle,cache,train,all")
    args = parser.parse_args()

    failures: List[str] = []
    for name in parse_selected(args.only):
        try:
            CHECKS[name]()
        except Exception as exc:
            print(f"{name.upper()}: FAIL - {type(exc).__name__}: {exc}")
            failures.append(name)

    if failures:
        print(f"OVERALL: FAIL - {', '.join(failures)}")
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
