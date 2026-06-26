"""Measure-only CPU/OOV evidence harness for tok3 hidden-domain candidates.

Loads generate_tokenizers.py-produced tok3 candidate pickles (it NEVER trains)
and writes per-(vocab, split) rows to tuning/results/tok3_cpu_sweep.csv with the
efficiency/speed metrics plus the checker-sensitive safety gates the hidden
domain relies on: zero [UNK] (byte fallback), reconstruction, >=1 bigram token,
0 trigram tokens, and an OOV stress pass-rate over deliberately unseen text.

Splits: domain_1_dev, domain_2_dev, combined_visible_dev, stress_oov.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from base_tokenizer import BaseTokenizer
from test_tokenizer import (
    calculate_efficiency,
    load_texts,
    test_reconstruction,
)


def measure_speed(tokenizer, texts, repeats):
    """tokens/sec via perf_counter (sub-us resolution, unlike time.time()'s ~15ms
    Windows granularity, which divides-by-zero on the tiny stress split). Mirrors
    test_tokenizer.measure_encoding_speed's contract with a zero-elapsed guard."""
    total_tokens = 0
    start = time.perf_counter()
    for _ in range(repeats):
        for text in texts:
            total_tokens += len(tokenizer.encode(text))
    elapsed = time.perf_counter() - start
    tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0.0

    single_tokens = 0
    single_start = time.perf_counter()
    for text in texts:
        single_tokens += len(tokenizer.encode(text))
    single_time = time.perf_counter() - single_start
    return tokens_per_second, single_time, single_tokens


# Deliberately unseen / cross-style text for OOV robustness. Covers unseen
# Unicode scripts, emoji, byte-fallback chars, URLs, @mentions, #hashtags,
# ALLCAPS, punctuation-heavy news, and mixed social/news strings.
STRESS_CASES = [
    "BREAKING: Markets tumbled 3.2% amid rate fears, the Fed said Tuesday.",
    "@user OMG this is sooo cool 😭🔥 #blessed #nofilter",
    "Visit https://example.com/path?q=1&x=2 for the full report.",
    "Σύμφωνα με την έκθεση, η οικονομία αναπτύχθηκε κατά 2%.",
    "東京の天δ気は今日晴れです。ニュース速報をお届けします。",
    "Цены на нефть выросли на 5 процентов за неделю.",
    "تقرير: ارتفعت الأسعار بنسبة ٪٥ خلال الأسبوع الماضي.",
    "u won't BELIEVE what happened next... 👀👇 (thread 1/9)",
    "The C.E.O. of A.B.C. Corp. resigned; shares fell —2.5%.",
    "naïve café résumé coördinate fiancé jalapeño piñata",
    "RT @news: \"quote\" — said the official, per Reuters.",
    "Mixed: BREAKING news 🚨 @POTUS says https://t.co/abc #live NOW!!!",
    "Ｆｕｌｌｗｉｄｔｈ ＡＢＣ and ½ ¾ ½ fractions ∑∫√ math.",
    "lol idk tbh ngl fr fr no cap 💀💀 that's wild",
    "Q3 EPS: $1.23 vs. $0.98 est.; revenue +12% YoY (GAAP).",
]

SPLIT_ORDER = ["domain_1_dev", "domain_2_dev", "combined_visible_dev", "stress_oov"]

CSV_FIELDS = [
    "tokenizer_index",
    "vocab_size",
    "tokenizer_path",
    "eval_split",
    "tokens_per_char",
    "tokens_per_second",
    "encode_seconds",
    "dev_chars",
    "dev_tokens",
    "reconstruction_rate",
    "unk_count",
    "bigram_token_count",
    "trigram_token_count",
    "oov_stress_cases_passed",
    "oov_stress_cases_total",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def candidate_path(candidate_dir: Path, vocab_size: int) -> Path:
    return candidate_dir / f"tokenizer_3_vocab{vocab_size}.pkl"


def count_bigram_trigram(tokenizer) -> tuple[int, int]:
    """Mirror check_submission.py: a token's surface = token.replace(space,' ').strip();
    exactly one internal space => bigram, two or more => trigram (must be 0)."""
    space = tokenizer.space_token
    specials = set(tokenizer.special_tokens)
    bigrams = 0
    trigrams = 0
    for token in tokenizer.token_to_id:
        if token in specials:
            continue
        spaces = token.replace(space, " ").strip().count(" ")
        if spaces == 1:
            bigrams += 1
        elif spaces >= 2:
            trigrams += 1
    return bigrams, trigrams


def unk_count_for_texts(tokenizer, texts) -> int:
    unk_id = tokenizer.special_tokens["[UNK]"]
    total = 0
    for text in texts:
        total += tokenizer.encode(text).count(unk_id)
    return total


def oov_stress_pass(tokenizer) -> tuple[int, int]:
    unk_id = tokenizer.special_tokens["[UNK]"]
    passed = 0
    for case in STRESS_CASES:
        ids = tokenizer.encode(case)
        recon = tokenizer.decode(ids)
        no_unk = unk_id not in ids
        reconstructs = recon.replace(" ", "") == case.replace(" ", "")
        if no_unk and reconstructs:
            passed += 1
    return passed, len(STRESS_CASES)


def measure_split(tokenizer, texts, repeats, reconstruction_samples) -> dict:
    tps, encode_seconds, dev_tokens = measure_speed(tokenizer, texts, repeats)
    tokens_per_char = calculate_efficiency(tokenizer, [], texts)
    reconstruction_rate = test_reconstruction(
        tokenizer, texts, sample_size=reconstruction_samples
    )
    dev_chars = sum(len(t) for t in texts)
    unk_count = unk_count_for_texts(tokenizer, texts)
    return {
        "tokens_per_char": f"{tokens_per_char:.8f}",
        "tokens_per_second": f"{tps:.4f}",
        "encode_seconds": f"{encode_seconds:.12f}",
        "dev_chars": str(dev_chars),
        "dev_tokens": str(dev_tokens),
        "reconstruction_rate": f"{reconstruction_rate:.6f}",
        "unk_count": str(unk_count),
    }


def rows_for_candidate(vocab_size, candidate_dir, splits, repeats, recon_samples):
    path = candidate_path(candidate_dir, vocab_size)
    if not path.exists():
        raise FileNotFoundError(f"missing tok3 candidate: {path}")
    tokenizer = BaseTokenizer.load(str(path))
    bigrams, trigrams = count_bigram_trigram(tokenizer)
    oov_passed, oov_total = oov_stress_pass(tokenizer)

    rows = []
    for split_name in SPLIT_ORDER:
        metrics = measure_split(
            tokenizer, splits[split_name], repeats, recon_samples
        )
        row = {
            "tokenizer_index": "3",
            "vocab_size": str(vocab_size),
            "tokenizer_path": str(path),
            "eval_split": split_name,
            "bigram_token_count": str(bigrams),
            "trigram_token_count": str(trigrams),
            "oov_stress_cases_passed": str(oov_passed),
            "oov_stress_cases_total": str(oov_total),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure tok3 candidate CPU/OOV evidence (no training)."
    )
    parser.add_argument(
        "--vocab_sizes", nargs="+", type=positive_int, default=[4000, 8000, 16000]
    )
    parser.add_argument("--candidate_dir", type=Path, default=Path("tuning/tokenizers"))
    parser.add_argument(
        "--results_csv", type=Path, default=Path("tuning/results/tok3_cpu_sweep.csv")
    )
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--reconstruction_samples", type=positive_int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = {
        "domain_1_dev": load_texts("data/domain_1_dev.txt"),
        "domain_2_dev": load_texts("data/domain_2_dev.txt"),
        "stress_oov": list(STRESS_CASES),
    }
    splits["combined_visible_dev"] = splits["domain_1_dev"] + splits["domain_2_dev"]

    all_rows = []
    for vocab_size in args.vocab_sizes:
        all_rows.extend(
            rows_for_candidate(
                vocab_size,
                args.candidate_dir,
                splits,
                args.repeats,
                args.reconstruction_samples,
            )
        )

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
