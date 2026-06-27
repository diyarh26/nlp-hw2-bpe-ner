"""Encode throughput benchmark + byte-identity harness for the encode optimizer.

Measure-only (like align_check.py / speed_check.py); nothing here ships in the
zip. Loads the three shipped tokenizers and, over many real dev lines:

  1. asserts the public encoder equals the slow apply-merges-in-order oracle on
     a small subset (the behavioral ground truth, kept small because the oracle
     replays *every* merge per line);
  2. asserts ``_apply_ranked_merges`` equals ``_reference_greedy`` -- a frozen
     copy of the original O(n^2) greedy merge loop -- on the FULL sample, so an
     encode-loop rewrite is proven byte-identical at scale (fast, greedy vs
     greedy);
  3. times pure ``_encode_uncached`` throughput (no cache) for the speedup story.

Usage:
    py -m uv run python code/encode_bench.py --lines 4000 --repeat 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from base_tokenizer import BaseTokenizer  # noqa: E402
from bpe_tokenizer import BPETokenizer  # noqa: E402


def read_lines(path: Path, limit: int) -> List[str]:
    out: List[str] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if text:
                out.append(text)
            if len(out) >= limit:
                break
    return out


def read_tagged_texts(path: Path, limit: int) -> List[str]:
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


def _reference_greedy(tok: BPETokenizer, symbols: List[str]) -> List[str]:
    """Frozen copy of the original O(n^2) greedy lowest-rank merge loop."""
    while True:
        best_rank = None
        best_index = None
        best_pair = None
        for index in range(len(symbols) - 1):
            pair = (symbols[index], symbols[index + 1])
            rank = tok.merge_ranks.get(pair)
            if rank is None or not tok._legal_merge(pair[0], pair[1]):
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
            + symbols[best_index + 2:]
        )
    return symbols


def slow_replay_encode(tok: BPETokenizer, text: str) -> List[int]:
    symbols = tok._text_to_encode_symbols(text)
    for pair in tok.merges:
        symbols = tok._apply_pair(symbols, pair)
    return tok._ids_from_symbols(symbols)


def load_tokenizer(name: str) -> BPETokenizer:
    return BaseTokenizer.load(str(ROOT_DIR / "trained_tokenizers" / name))


def sample_lines(lines: int) -> List[str]:
    half = max(1, lines // 2)
    quarter = max(1, lines // 4)
    samples: List[str] = []
    samples += read_lines(ROOT_DIR / "data" / "domain_1_dev.txt", half)
    samples += read_lines(ROOT_DIR / "data" / "domain_2_dev.txt", half)
    samples += read_tagged_texts(ROOT_DIR / "data" / "ner_data" / "dev_1_binary.tagged", quarter)
    samples += read_tagged_texts(ROOT_DIR / "data" / "ner_data" / "dev_2_binary.tagged", quarter)
    return samples


def check_oracle(tok: BPETokenizer, name: str, samples: List[str]) -> int:
    bad = 0
    for text in samples:
        tok._encode_cache.clear()
        if tok.encode(text) != slow_replay_encode(tok, text):
            bad += 1
            if bad <= 3:
                print(f"  [ORACLE MISMATCH] {name}: {text[:60]!r}")
    return bad


def check_reference(tok: BPETokenizer, name: str, samples: List[str]) -> int:
    bad = 0
    for text in samples:
        symbols = tok._text_to_encode_symbols(text)
        got = tok._apply_ranked_merges(list(symbols))
        want = _reference_greedy(tok, list(symbols))
        if got != want:
            bad += 1
            if bad <= 3:
                print(f"  [REF MISMATCH] {name}: {text[:60]!r}")
    return bad


def bench_uncached(tok: BPETokenizer, samples: List[str], repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        tok._encode_cache.clear()
        start = time.perf_counter()
        for text in samples:
            tok._encode_uncached(text)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode throughput + byte-identity.")
    parser.add_argument("--lines", type=int, default=4000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--oracle-lines", type=int, default=300)
    args = parser.parse_args()

    samples = sample_lines(args.lines)
    total_chars = sum(len(s) for s in samples)
    oracle_subset = samples[: args.oracle_lines] + samples[-args.oracle_lines:]
    print(f"samples={len(samples)} total_chars={total_chars} oracle_subset={len(oracle_subset)}")

    failures = 0
    for name in ("tokenizer_1.pkl", "tokenizer_2.pkl", "tokenizer_3.pkl"):
        tok = load_tokenizer(name)
        oracle_bad = check_oracle(tok, name, oracle_subset)
        ref_bad = check_reference(tok, name, samples)
        if oracle_bad or ref_bad:
            failures += 1
            print(f"{name}: FAIL - oracle_mismatch={oracle_bad} ref_mismatch={ref_bad}")
            continue
        seconds = bench_uncached(tok, samples, args.repeat)
        cps = total_chars / seconds if seconds else 0.0
        lps = len(samples) / seconds if seconds else 0.0
        print(
            f"{name}: OK oracle+ref | uncached best={seconds:.4f}s "
            f"| {cps:,.0f} chars/s | {lps:,.0f} lines/s"
        )

    if failures:
        print(f"OVERALL: FAIL - {failures} tokenizer(s) mismatched")
        return 1
    print("OVERALL: PASS - encode byte-identical to oracle and reference greedy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
