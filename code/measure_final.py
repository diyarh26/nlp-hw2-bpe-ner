"""Phase 5 final competition measurement for the SHIPPING tokenizers.

Measure-only (this NEVER trains). It loads the three staged deliverables
`trained_tokenizers/tokenizer_{1,2,3}.pkl`, asserts each is byte-identical
(SHA-256) to the candidate that was scored for NER F1 on the Tesla M60 VM, and
then measures the three equally-weighted competition axes on the dev files:

  - efficiency:  tokens/char  AND  tokens/sentence (non-empty lines)
  - speed:       cold single-pass tokens/sec (cache empty at start, warms
                 naturally during the pass -- the honest "encode the test set
                 once" number) AND the provided reference harness number
                 (test_tokenizer.measure_encoding_speed, repeats=5)
  - F1:          attached from the byte-identical VM run (EVAL-02/03/04 evidence)

plus the checker-sensitive safety gates the submission depends on: zero [UNK]
(byte fallback), reconstruction rate, >=1 bigram token, and 0 trigram tokens.

Because the staged pkl's SHA-256 must equal the VM-scored candidate's SHA-256
(hard assert below), the reused F1 provably belongs to the exact shipping file.
Writes one consolidated, report-ready table to tuning/results/final_eval.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

from base_tokenizer import BaseTokenizer
from test_tokenizer import load_texts, measure_encoding_speed, test_reconstruction


# Final-submission provenance. Each entry ties a shipping pkl to the
# byte-identical VM candidate and the per-domain VM F1 already captured for it
# (source CSV recorded for traceability). The SHA-256 equality is enforced at
# runtime, so if a shipping pkl ever drifts from its scored candidate the run
# fails loudly instead of attaching a stale F1.
FINAL = [
    {
        # tok1 finalized on the balanced mix @ vocab 6000 (tok1-opt Batch 1: best
        # Twitter F1 0.4780, +0.043 over the prior vocab-8000 domain-1-only tok1).
        # Structured pre-tok (Batch 2) was tried and reverted (it hurt F1).
        "index": 1,
        "vocab_size": 6000,
        "shipping": "trained_tokenizers/tokenizer_1.pkl",
        "candidate": "tuning/tok1_sweep/mix_v6000.pkl",
        "evals": [
            {
                "split": "domain_1_dev",
                "dev_file": "data/domain_1_dev.txt",
                "vm_f1": 0.4780,
                "vm_f1_source": "tuning/results/tok1_sweep_f1.csv",
            },
        ],
    },
    {
        "index": 2,
        "vocab_size": 16000,
        "shipping": "trained_tokenizers/tokenizer_2.pkl",
        "candidate": "tuning/tokenizers/tokenizer_2_vocab16000.pkl",
        "evals": [
            {
                "split": "domain_2_dev",
                "dev_file": "data/domain_2_dev.txt",
                "vm_f1": 0.9557,
                "vm_f1_source": "tuning/results/vm_f1_results.csv",
            },
        ],
    },
    {
        "index": 3,
        "vocab_size": 8000,
        "shipping": "trained_tokenizers/tokenizer_3.pkl",
        "candidate": "tuning/tokenizers/tokenizer_3_vocab8000.pkl",
        # tok3 generalizes to a hidden domain -> report it on BOTH visible domains.
        "evals": [
            {
                "split": "domain_1_dev",
                "dev_file": "data/domain_1_dev.txt",
                "vm_f1": 0.4217,
                "vm_f1_source": "tuning/results/tok3_vm_f1_results.csv",
            },
            {
                "split": "domain_2_dev",
                "dev_file": "data/domain_2_dev.txt",
                "vm_f1": 0.9587,
                "vm_f1_source": "tuning/results/tok3_vm_f1_results.csv",
            },
        ],
    },
]

CSV_FIELDS = [
    "tokenizer_index",
    "vocab_size",
    "sha256",
    "shipping_path",
    "eval_split",
    "dev_file",
    "vm_f1",
    "vm_f1_source",
    "tokens_per_char",
    "tokens_per_sentence",
    "tokens_per_second_cold",
    "cold_pass_seconds",
    "tokens_per_second_ref5",
    "dev_chars",
    "dev_tokens",
    "dev_sentences",
    "reconstruction_rate",
    "unk_count",
    "bigram_token_count",
    "trigram_token_count",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def measure_cold(path: str, texts: list[str]) -> dict:
    """One cold pass over `texts` on a freshly loaded tokenizer (empty encode
    cache). This is the honest "encode the test set once" speed and the
    pass-invariant efficiency counts. Returns the raw metrics for one split."""
    tokenizer = BaseTokenizer.load(path)
    unk_id = tokenizer.special_tokens["[UNK]"]
    total_tokens = 0
    total_chars = 0
    unk_count = 0
    start = time.perf_counter()
    for text in texts:
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        total_chars += len(text)
        unk_count += ids.count(unk_id)
    elapsed = time.perf_counter() - start

    n_sentences = sum(1 for t in texts if t)
    tps_cold = total_tokens / elapsed if elapsed > 0 else 0.0
    return {
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "n_sentences": n_sentences,
        "cold_seconds": elapsed,
        "tps_cold": tps_cold,
        "unk_count": unk_count,
        "tokens_per_char": total_tokens / total_chars if total_chars else 0.0,
        "tokens_per_sentence": total_tokens / n_sentences if n_sentences else 0.0,
    }


def row_for_eval(entry: dict, eval_spec: dict, sha: str, repeats: int,
                 recon_samples: int) -> dict:
    shipping = entry["shipping"]
    texts = load_texts(eval_spec["dev_file"])

    cold = measure_cold(shipping, texts)

    # Provided reference-harness number: fresh load so its 5 passes start cold,
    # exactly as test_tokenizer would measure it standalone.
    ref_tok = BaseTokenizer.load(shipping)
    tps_ref5, _, _ = measure_encoding_speed(ref_tok, texts, repeats=repeats)

    recon = test_reconstruction(ref_tok, texts, sample_size=recon_samples)
    bigrams, trigrams = count_bigram_trigram(ref_tok)

    return {
        "tokenizer_index": str(entry["index"]),
        "vocab_size": str(entry["vocab_size"]),
        "sha256": sha,
        "shipping_path": shipping,
        "eval_split": eval_spec["split"],
        "dev_file": eval_spec["dev_file"],
        "vm_f1": f"{eval_spec['vm_f1']:.4f}",
        "vm_f1_source": eval_spec["vm_f1_source"],
        "tokens_per_char": f"{cold['tokens_per_char']:.8f}",
        "tokens_per_sentence": f"{cold['tokens_per_sentence']:.6f}",
        "tokens_per_second_cold": f"{cold['tps_cold']:.4f}",
        "cold_pass_seconds": f"{cold['cold_seconds']:.6f}",
        "tokens_per_second_ref5": f"{tps_ref5:.4f}",
        "dev_chars": str(cold["total_chars"]),
        "dev_tokens": str(cold["total_tokens"]),
        "dev_sentences": str(cold["n_sentences"]),
        "reconstruction_rate": f"{recon:.6f}",
        "unk_count": str(cold["unk_count"]),
        "bigram_token_count": str(bigrams),
        "trigram_token_count": str(trigrams),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 final competition measurement (measure-only)."
    )
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=Path("tuning/results/final_eval.csv"),
    )
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--reconstruction_samples", type=positive_int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for entry in FINAL:
        shipping = Path(entry["shipping"])
        candidate = Path(entry["candidate"])
        if not shipping.exists():
            raise FileNotFoundError(f"missing shipping pkl: {shipping}")
        if not candidate.exists():
            raise FileNotFoundError(f"missing VM candidate pkl: {candidate}")

        ship_sha = sha256_of(shipping)
        cand_sha = sha256_of(candidate)
        if ship_sha != cand_sha:
            raise SystemExit(
                f"SHA-256 MISMATCH for tokenizer_{entry['index']}: shipping "
                f"{shipping} ({ship_sha[:12]}) != VM candidate {candidate} "
                f"({cand_sha[:12]}). The reused VM F1 would NOT apply -- refusing "
                f"to attach a stale F1. Re-stage or re-score before measuring."
            )
        print(
            f"tokenizer_{entry['index']}: SHA-256 {ship_sha[:12]} matches VM "
            f"candidate -> reused F1 is valid."
        )

        for eval_spec in entry["evals"]:
            row = row_for_eval(
                entry, eval_spec, ship_sha, args.repeats,
                args.reconstruction_samples,
            )
            print(
                f"  {row['eval_split']}: F1={row['vm_f1']} "
                f"tok/char={row['tokens_per_char']} "
                f"tok/sent={row['tokens_per_sentence']} "
                f"cold_tok/s={row['tokens_per_second_cold']} "
                f"unk={row['unk_count']} bigrams={row['bigram_token_count']} "
                f"trigrams={row['trigram_token_count']}"
            )
            rows.append(row)

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
