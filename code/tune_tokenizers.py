"""Measure generated tokenizer candidates for Phase 3 tuning.

This harness intentionally does not train tokenizers. Candidate pickles must be
created by generate_tokenizers.py, then copied into tuning/tokenizers/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from base_tokenizer import BaseTokenizer
from test_tokenizer import (
    calculate_efficiency,
    load_texts,
    measure_encoding_speed,
    test_reconstruction,
)


DOMAIN_FILES = {
    1: {
        "train_file": Path("data/domain_1_train.txt"),
        "dev_file": Path("data/domain_1_dev.txt"),
    },
    2: {
        "train_file": Path("data/domain_2_train.txt"),
        "dev_file": Path("data/domain_2_dev.txt"),
    },
}


CSV_FIELDS = [
    "domain",
    "tokenizer_index",
    "vocab_size",
    "tokenizer_path",
    "train_file",
    "dev_file",
    "tokens_per_char",
    "tokens_per_second",
    "encode_seconds",
    "dev_chars",
    "dev_tokens",
    "reconstruction_rate",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def candidate_path(candidate_dir: Path, tokenizer_index: int, vocab_size: int) -> Path:
    return candidate_dir / f"tokenizer_{tokenizer_index}_vocab{vocab_size}.pkl"


def measure_candidate(
    tokenizer_index: int,
    vocab_size: int,
    candidate_dir: Path,
    repeats: int,
    reconstruction_samples: int,
) -> dict[str, str]:
    paths = DOMAIN_FILES[tokenizer_index]
    tokenizer_file = candidate_path(candidate_dir, tokenizer_index, vocab_size)
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"missing candidate tokenizer: {tokenizer_file}")

    tokenizer = BaseTokenizer.load(str(tokenizer_file))
    train_texts = load_texts(str(paths["train_file"]))
    dev_texts = load_texts(str(paths["dev_file"]))

    tokens_per_char = calculate_efficiency(tokenizer, train_texts, dev_texts)
    tokens_per_second, encode_seconds, dev_tokens = measure_encoding_speed(
        tokenizer, dev_texts, repeats=repeats
    )
    reconstruction_rate = test_reconstruction(
        tokenizer, dev_texts, sample_size=reconstruction_samples
    )
    dev_chars = sum(len(text) for text in dev_texts)

    return {
        "domain": str(tokenizer_index),
        "tokenizer_index": str(tokenizer_index),
        "vocab_size": str(vocab_size),
        "tokenizer_path": str(tokenizer_file),
        "train_file": str(paths["train_file"]),
        "dev_file": str(paths["dev_file"]),
        "tokens_per_char": f"{tokens_per_char:.8f}",
        "tokens_per_second": f"{tokens_per_second:.4f}",
        "encode_seconds": f"{encode_seconds:.12f}",
        "dev_chars": str(dev_chars),
        "dev_tokens": str(dev_tokens),
        "reconstruction_rate": f"{reconstruction_rate:.6f}",
    }


def write_sweep(
    vocab_sizes: Iterable[int],
    candidate_dir: Path,
    results_csv: Path,
    repeats: int,
    reconstruction_samples: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for vocab_size in vocab_sizes:
        for tokenizer_index in (1, 2):
            rows.append(
                measure_candidate(
                    tokenizer_index=tokenizer_index,
                    vocab_size=vocab_size,
                    candidate_dir=candidate_dir,
                    repeats=repeats,
                    reconstruction_samples=reconstruction_samples,
                )
            )

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure generate_tokenizers.py-produced tok1/tok2 candidates."
    )
    parser.add_argument(
        "--vocab_sizes",
        nargs="+",
        type=positive_int,
        default=[4000, 8000, 16000],
        help="Vocabulary sizes to measure.",
    )
    parser.add_argument(
        "--candidate_dir",
        type=Path,
        default=Path("tuning/tokenizers"),
        help="Directory containing tokenizer_{1,2}_vocabV.pkl candidates.",
    )
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=Path("tuning/results/cpu_sweep.csv"),
        help="CSV path for CPU sweep metrics.",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=5,
        help="Encoding speed repeats passed to test_tokenizer helpers.",
    )
    parser.add_argument(
        "--reconstruction_samples",
        type=positive_int,
        default=10,
        help="Number of dev texts sampled for decode reconstruction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = write_sweep(
        vocab_sizes=args.vocab_sizes,
        candidate_dir=args.candidate_dir,
        results_csv=args.results_csv,
        repeats=args.repeats,
        reconstruction_samples=args.reconstruction_samples,
    )
    print(f"Wrote {len(rows)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
