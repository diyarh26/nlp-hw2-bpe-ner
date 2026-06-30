"""Recipe sweep for tok1 (Twitter) and tok3 (hidden-domain) candidates.

The data-recipe lever was barely explored before this round: only pure-domain-1,
the 1:1 balanced mix, and both-full-corpora were ever F1-scored. This sweep adds
the two unexplored, promising regions:

  * pure domain-1 at HIGHER vocab (the pure-d1 F1 curve was still climbing at
    v10000=0.4633, and bigger vocab also wins the efficiency + speed axes), and
  * domain-1-HEAVY mixes (2:1, 3:1 d1:d2) between the 1:1 balanced mix (best so
    far, 0.4780 @ 6000) and pure-d1 -- keeping more Twitter coverage while
    retaining the news-data regularization that made the mix beat pure-d1.

Each candidate is trained EXACTLY as generate_tokenizers.py trains it: a fresh
`BPETokenizer(vocab_size=V)` over the train file read with `readlines()`
(newlines kept), saved with `.save()`. Training is deterministic and
slot-independent, so a candidate produced here is byte-identical to a stock
generator run with the same file/vocab. The provenance gate PROVES this: it first
re-trains the 1:1 mix @ 6000 and asserts the bytes match the shipped tok1
(ef7db8c2...); if that fails it aborts rather than sweep an unreproducible
baseline.

For each candidate it records the CPU competition axes on BOTH visible dev sets
(tokens/char, tokens/sentence, honest cold tokens/sec) plus the checker-sensitive
safety gates (zero [UNK], reconstruction, >=1 bigram, 0 trigram). NER F1 itself is
measured later on the VM via the generated runbook.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

from base_tokenizer import BaseTokenizer
from bpe_tokenizer import BPETokenizer
from test_tokenizer import load_texts, test_reconstruction


# Shipped tok1/tok3 SHA-256 (1:1 balanced mix @ 6000). The provenance gate below
# must reproduce this from data/tok3_mix_d1sample_d2all_seed42.txt @ 6000.
SHIPPED_MIX_V6000_SHA = (
    "ef7db8c2382f3c7d7a48828d8ae1da5007ff0c65d746fa3a248fae37d673db14"
)

# variant, train_file (read like the generator), vocab, role
CANDIDATES = [
    # pure domain-1, vocab extension (Twitter-specialized; better efficiency)
    ("d1_v8000", "data/domain_1_train.txt", 8000, "tok1"),
    ("d1_v12000", "data/domain_1_train.txt", 12000, "tok1"),
    ("d1_v16000", "data/domain_1_train.txt", 16000, "tok1"),
    # domain-1-heavy mixes (also tok3 generalist candidates -> scored on both dev sets)
    ("mix2_v5500", "data/mix_d1x2_d2all_seed42.txt", 5500, "tok1+tok3"),
    ("mix2_v6000", "data/mix_d1x2_d2all_seed42.txt", 6000, "tok1+tok3"),
    ("mix2_v6500", "data/mix_d1x2_d2all_seed42.txt", 6500, "tok1+tok3"),
    ("mix3_v6000", "data/mix_d1x3_d2all_seed42.txt", 6000, "tok1+tok3"),
]

CSV_FIELDS = [
    "variant", "vocab_size", "train_file", "role", "sha256", "candidate_path",
    "eval_split", "eval_file", "tokens_per_char", "tokens_per_sentence",
    "tokens_per_second_cold", "cold_pass_seconds", "dev_chars", "dev_tokens",
    "dev_sentences", "reconstruction_rate", "unk_count", "bigram_token_count",
    "trigram_token_count",
]

EVAL_SPLITS = [
    ("domain_1_dev", "data/domain_1_dev.txt"),
    ("domain_2_dev", "data/domain_2_dev.txt"),
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_like_generator(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        return fh.readlines()


def count_bigram_trigram(tokenizer) -> tuple[int, int]:
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


def train_candidate(texts: list[str], vocab: int, out_path: Path) -> str:
    tok = BPETokenizer(vocab_size=vocab)
    tok.train(texts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return sha256_of(out_path)


def measure_cold(path: Path, texts: list[str], recon_samples: int) -> dict:
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
    recon = test_reconstruction(tok, texts, sample_size=recon_samples)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tok1/tok3 recipe sweep (CPU; F1 on VM later).")
    p.add_argument("--gen_dir", type=Path, default=Path("tuning/recipe_sweep"))
    p.add_argument("--results_csv", type=Path,
                   default=Path("tuning/results/recipe2_cpu.csv"))
    p.add_argument("--mix_v6000", type=Path,
                   default=Path("data/tok3_mix_d1sample_d2all_seed42.txt"))
    p.add_argument("--recon_samples", type=int, default=3000)
    p.add_argument("--skip_provenance", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.gen_dir.mkdir(parents=True, exist_ok=True)

    # --- Provenance gate: reproduce the shipped tok1 (1:1 mix @ 6000) ---
    if not args.skip_provenance:
        print(f"[provenance] re-training 1:1 mix @ 6000 to match shipped tok1 "
              f"({SHIPPED_MIX_V6000_SHA[:12]}) ...")
        mix_texts = read_like_generator(args.mix_v6000)
        got = train_candidate(mix_texts, 6000, args.gen_dir / "_provenance_mix_v6000.pkl")
        if got != SHIPPED_MIX_V6000_SHA:
            raise SystemExit(
                f"PROVENANCE MISMATCH: re-trained 1:1 mix @ 6000 -> {got[:12]} but "
                f"shipped tok1 is {SHIPPED_MIX_V6000_SHA[:12]}. Refusing to sweep an "
                f"unreproducible baseline -- diagnose data/line-handling first."
            )
        print(f"[provenance] OK: {got[:12]} matches shipped tok1.")

    eval_texts = {split: load_texts(path) for split, path in EVAL_SPLITS}
    for split, path in EVAL_SPLITS:
        print(f"  eval {split}: {len(eval_texts[split])} lines ({path})")

    read_cache: dict[str, list[str]] = {}
    rows = []
    for variant, train_file, vocab, role in CANDIDATES:
        out = args.gen_dir / f"{variant}.pkl"
        if out.exists():
            sha = sha256_of(out)
            print(f"\n[{variant}] reuse existing pkl ({sha[:12]})")
        else:
            if train_file not in read_cache:
                print(f"\n[{variant}] reading {train_file} ...")
                read_cache[train_file] = read_like_generator(Path(train_file))
            print(f"[{variant}] training vocab {vocab} on {train_file} "
                  f"({len(read_cache[train_file])} lines) -> {out}")
            t0 = time.perf_counter()
            sha = train_candidate(read_cache[train_file], vocab, out)
            print(f"[{variant}] trained in {time.perf_counter()-t0:.1f}s, sha {sha[:12]}")

        for split, _path in EVAL_SPLITS:
            m = measure_cold(out, eval_texts[split], args.recon_samples)
            row = {
                "variant": variant, "vocab_size": str(vocab),
                "train_file": train_file, "role": role, "sha256": sha,
                "candidate_path": str(out), "eval_split": split,
                "eval_file": _path,
            }
            for k in ("tokens_per_char", "tokens_per_sentence",
                      "tokens_per_second_cold", "cold_pass_seconds",
                      "reconstruction_rate"):
                row[k] = (f"{m[k]:.8f}" if "per_char" in k or "recon" in k
                          else f"{m[k]:.6f}")
            row.update({
                "dev_chars": str(m["dev_chars"]), "dev_tokens": str(m["dev_tokens"]),
                "dev_sentences": str(m["dev_sentences"]),
                "unk_count": str(m["unk_count"]),
                "bigram_token_count": str(m["bigram_token_count"]),
                "trigram_token_count": str(m["trigram_token_count"]),
            })
            rows.append(row)
            print(f"    {split}: tok/char={row['tokens_per_char']} "
                  f"tok/sent={row['tokens_per_sentence']} "
                  f"cold_tok/s={row['tokens_per_second_cold']} "
                  f"unk={row['unk_count']} big={row['bigram_token_count']} "
                  f"tri={row['trigram_token_count']} recon={row['reconstruction_rate']}")

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.results_csv}")
    print(f"Candidate pkls in {args.gen_dir}/ -> push these for the VM F1 sweep.")


if __name__ == "__main__":
    main()
