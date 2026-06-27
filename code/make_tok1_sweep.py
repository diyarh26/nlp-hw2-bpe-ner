"""Batch 1.5: tok1 balanced-mix vocab sweep (Twitter F1 optimization).

Batch 1 found the balanced domain-1+2 mix is the best DATA recipe for tokenizer 1
(Twitter), scoring F1 0.4780 at vocab 6000 -- above domain-1-only and far above
both-full-corpora -- but we have only ONE vocab point on it. This helper sweeps
the mix across more vocab sizes to map its F1/efficiency curve and see whether it
reaches the ~0.50 target without touching the core tokenizer.

It trains each candidate exactly the way generate_tokenizers.py does -- a fresh
`BPETokenizer(vocab_size=V)` over the mix lines read with `readlines()` (newlines
kept, NOT stripped), saved via `tokenizer.save()` -- so a candidate is byte-
identical to a stock generator run with the mix in any slot (training is
deterministic and slot-independent). To PROVE that equivalence, it first
re-trains vocab 6000 and asserts the bytes match the VM-scored
`tuning/tok1_exp/mix_v6000.pkl` (ef7db8c2...). If that provenance gate fails it
aborts rather than sweep an unreproducible baseline.

For each candidate it also records the CPU competition axes on the Twitter dev
set (tokens/char, tokens/sentence, honest cold-pass tokens/sec) plus the
checker-sensitive safety gates (zero [UNK], reconstruction, >=1 bigram, 0
trigram). F1 itself is measured later on the VM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import time
from pathlib import Path

import numpy as np

from base_tokenizer import BaseTokenizer
from bpe_tokenizer import BPETokenizer
from test_tokenizer import test_reconstruction


def set_seed(seed: int) -> None:
    """Mirror generate_tokenizers.set_seed (harmless: training uses no RNG)."""
    random.seed(seed)
    np.random.seed(seed)


def read_lines_like_generator(path: Path) -> list[str]:
    """Exactly generate_tokenizers.read_text_file: readlines(), newlines kept."""
    with path.open("r", encoding="utf-8") as fh:
        return fh.readlines()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count_bigram_trigram(tokenizer) -> tuple[int, int]:
    """Mirror check_submission.py bigram/trigram surface test."""
    space = tokenizer.space_token
    specials = set(tokenizer.special_tokens)
    bigrams = trigrams = 0
    for token in tokenizer.token_to_id:
        if token in specials:
            continue
        spaces = token.replace(space, " ").strip().count(" ")
        if spaces == 1:
            bigrams += 1
        elif spaces >= 2:
            trigrams += 1
    return bigrams, trigrams


def train_mix(texts: list[str], vocab_size: int, seed: int, out_path: Path) -> str:
    set_seed(seed)
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(texts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return sha256_of(out_path)


def measure_cold(path: Path, texts: list[str]) -> dict:
    """One cold pass (fresh load, empty cache) over the Twitter dev set."""
    tok = BaseTokenizer.load(str(path))
    unk_id = tok.special_tokens["[UNK]"]
    total_tokens = total_chars = unk = 0
    start = time.perf_counter()
    for t in texts:
        ids = tok.encode(t)
        total_tokens += len(ids)
        total_chars += len(t)
        unk += ids.count(unk_id)
    elapsed = time.perf_counter() - start
    n_sent = sum(1 for t in texts if t)
    bigrams, trigrams = count_bigram_trigram(tok)
    recon = test_reconstruction(tok, texts, sample_size=5000)
    return {
        "tokens_per_char": total_tokens / total_chars if total_chars else 0.0,
        "tokens_per_sentence": total_tokens / n_sent if n_sent else 0.0,
        "tokens_per_second_cold": total_tokens / elapsed if elapsed > 0 else 0.0,
        "cold_pass_seconds": elapsed,
        "dev_chars": total_chars,
        "dev_tokens": total_tokens,
        "dev_sentences": n_sent,
        "reconstruction_rate": recon,
        "unk_count": unk,
        "bigram_token_count": bigrams,
        "trigram_token_count": trigrams,
    }


CSV_FIELDS = [
    "variant", "vocab_size", "train_data", "sha256", "candidate_path",
    "eval_file", "tokens_per_char", "tokens_per_sentence",
    "tokens_per_second_cold", "cold_pass_seconds", "dev_chars", "dev_tokens",
    "dev_sentences", "reconstruction_rate", "unk_count", "bigram_token_count",
    "trigram_token_count",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tok1 balanced-mix vocab sweep (Batch 1.5).")
    p.add_argument("--mix_file", type=Path,
                   default=Path("data/tok3_mix_d1sample_d2all_seed42.txt"))
    p.add_argument("--vocab_sizes", nargs="*", type=positive_int,
                   default=[7000, 8000, 10000, 12000])
    p.add_argument("--seed", type=positive_int, default=42)
    p.add_argument("--eval_file", type=Path, default=Path("data/domain_1_dev.txt"))
    p.add_argument("--gen_dir", type=Path, default=Path("tuning/_tok1_sweep"))
    p.add_argument("--vendor_dir", type=Path, default=Path("tuning/tok1_sweep"))
    p.add_argument("--results_csv", type=Path,
                   default=Path("tuning/results/tok1_sweep_cpu.csv"))
    p.add_argument("--check_vocab", type=positive_int, default=6000)
    p.add_argument("--check_pkl", type=Path,
                   default=Path("tuning/tok1_exp/mix_v6000.pkl"))
    p.add_argument("--skip_check", action="store_true")
    p.add_argument("--only_check", action="store_true",
                   help="Run only the provenance gate (reproduce mix_v6000) and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mix_file.exists():
        raise FileNotFoundError(f"mix file not found: {args.mix_file}")
    print(f"Reading mix lines (readlines, newlines kept) from {args.mix_file} ...")
    texts = read_lines_like_generator(args.mix_file)
    print(f"  {len(texts)} lines")

    # --- Provenance gate: reproduce the VM-scored mix_v6000 (ef7db8c2...) ---
    if not args.skip_check:
        check_out = args.gen_dir / f"check_mix_v{args.check_vocab}.pkl"
        print(f"\n[provenance] training mix @ vocab {args.check_vocab} to match "
              f"{args.check_pkl} ...")
        got = train_mix(texts, args.check_vocab, args.seed, check_out)
        if not args.check_pkl.exists():
            raise FileNotFoundError(f"reference pkl missing: {args.check_pkl}")
        want = sha256_of(args.check_pkl)
        if got != want:
            raise SystemExit(
                f"PROVENANCE MISMATCH: retrained mix @ {args.check_vocab} -> "
                f"{got[:12]} but {args.check_pkl} is {want[:12]}. The VM-scored "
                f"mix_v6000 is NOT reproduced by this method -- refusing to sweep "
                f"an unreproducible baseline. Diagnose line-handling/data first."
            )
        print(f"[provenance] OK: {got[:12]} matches the VM-scored mix_v6000.")
        if args.only_check:
            print("only_check set -> exiting after provenance gate.")
            return

    eval_texts = [line.strip() for line in args.eval_file.read_text(
        encoding="utf-8").splitlines()]
    print(f"\nTwitter dev eval set: {args.eval_file} ({len(eval_texts)} lines)")

    args.vendor_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # Always include the 6000 anchor in the results table for a complete curve.
    sweep = sorted(set(args.vocab_sizes) | {args.check_vocab})
    for vocab in sweep:
        out = args.gen_dir / f"mix_v{vocab}.pkl"
        if not out.exists():
            print(f"\nTraining mix @ vocab {vocab} -> {out}")
            sha = train_mix(texts, vocab, args.seed, out)
        else:
            sha = sha256_of(out)
            print(f"\nReusing existing {out} ({sha[:12]})")
        vendor = args.vendor_dir / f"mix_v{vocab}.pkl"
        vendor.write_bytes(out.read_bytes())

        metrics = measure_cold(out, eval_texts)
        row = {
            "variant": f"mix_v{vocab}",
            "vocab_size": str(vocab),
            "train_data": "balanced-mix",
            "sha256": sha,
            "candidate_path": str(vendor),
            "eval_file": str(args.eval_file),
        }
        for key in (
            "tokens_per_char", "tokens_per_sentence", "tokens_per_second_cold",
            "cold_pass_seconds", "reconstruction_rate",
        ):
            row[key] = f"{metrics[key]:.8f}" if "per_char" in key or "recon" in key \
                else f"{metrics[key]:.6f}"
        row.update({
            "dev_chars": str(metrics["dev_chars"]),
            "dev_tokens": str(metrics["dev_tokens"]),
            "dev_sentences": str(metrics["dev_sentences"]),
            "unk_count": str(metrics["unk_count"]),
            "bigram_token_count": str(metrics["bigram_token_count"]),
            "trigram_token_count": str(metrics["trigram_token_count"]),
        })
        rows.append(row)
        print(f"  vocab {vocab}: tok/char={row['tokens_per_char']} "
              f"tok/sent={row['tokens_per_sentence']} "
              f"cold_tok/s={row['tokens_per_second_cold']} "
              f"unk={row['unk_count']} bigrams={row['bigram_token_count']} "
              f"trigrams={row['trigram_token_count']} recon={row['reconstruction_rate']}")

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.results_csv}")
    print(f"Vendored candidates -> {args.vendor_dir}/ (push these for the VM F1 sweep)")


if __name__ == "__main__":
    main()
