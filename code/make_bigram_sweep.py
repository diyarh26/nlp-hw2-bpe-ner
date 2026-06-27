"""Lever 2 candidate generator + CPU efficiency sweep (measure-only NER-wise).

Trains tok1 (Twitter, the F1-sensitive one) on the balanced mix at vocab 6000
for several EXTRA_CROSS_WORD_BIGRAMS values, then measures the efficiency axis
(tokens/char, tokens/sentence) and the safety gates (reconstruction, [UNK]
count, bigram/trigram token counts) on domain_1_dev. K=0 is a provenance control
that MUST reproduce the shipped tok1 hash, proving the refactor left the baseline
byte-identical. F1 is NOT measured here (GPU/VM only); this just sizes the
efficiency win so the VM F1 sweep is worth running.

Usage:
    py -m uv run python code/make_bigram_sweep.py --ks 0 8 20 40 \
        --train_file data/tok3_mix_d1sample_d2all_seed42.txt \
        --dev_file data/domain_1_dev.txt --vocab 6000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import bpe_tokenizer  # noqa: E402
from bpe_tokenizer import BPETokenizer  # noqa: E402
from base_tokenizer import BaseTokenizer  # noqa: E402

SHIPPED_TOK1_SHA = "ef7db8c2382f3c7d7a48828d8ae1da5007ff0c65d746fa3a248fae37d673db14"


def read_train_lines(path: Path) -> list[str]:
    """Match generate_tokenizers.read_text_file EXACTLY (readlines keeps the
    trailing newline), so the K=0 candidate reproduces the shipped tok1 hash."""
    with path.open("r", encoding="utf-8") as fh:
        return fh.readlines()


def read_dev_lines(path: Path) -> list[str]:
    """Dev text is encoded the way the NER/checker call it: without newlines."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return [ln.rstrip("\n") for ln in fh]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bigram_trigram(tok: BPETokenizer) -> tuple[int, int]:
    space = tok.space_token
    specials = set(tok.special_tokens)
    big = tri = 0
    for token in tok.token_to_id:
        if token in specials:
            continue
        spaces = token.replace(space, " ").strip().count(" ")
        if spaces == 1:
            big += 1
        elif spaces >= 2:
            tri += 1
    return big, tri


def measure(tok: BPETokenizer, dev_lines: list[str]) -> dict:
    unk_id = tok.special_tokens["[UNK]"]
    total_tokens = total_chars = n_sent = unk = recon_ok = recon_n = 0
    for text in dev_lines:
        if not text:
            continue
        ids = tok.encode(text)
        total_tokens += len(ids)
        total_chars += len(text)
        n_sent += 1
        unk += ids.count(unk_id)
        if recon_n < 5000:
            recon_n += 1
            if tok.decode(ids).replace(" ", "") == text.replace(" ", ""):
                recon_ok += 1
    big, tri = bigram_trigram(tok)
    return {
        "tokens_per_char": total_tokens / total_chars if total_chars else 0.0,
        "tokens_per_sentence": total_tokens / n_sent if n_sent else 0.0,
        "dev_tokens": total_tokens,
        "dev_sentences": n_sent,
        "unk_count": unk,
        "reconstruction_rate": recon_ok / recon_n if recon_n else 0.0,
        "bigram_token_count": big,
        "trigram_token_count": tri,
        "vocab_size": tok.get_vocab_size(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 8, 20, 40])
    ap.add_argument("--train_file", default="data/tok3_mix_d1sample_d2all_seed42.txt")
    ap.add_argument("--dev_file", default="data/domain_1_dev.txt")
    ap.add_argument("--vocab", type=int, default=6000)
    ap.add_argument("--out_dir", default="tuning/bigram_sweep")
    ap.add_argument("--results_csv", default="tuning/results/bigram_sweep_cpu.csv")
    args = ap.parse_args()

    train_lines = read_train_lines(ROOT_DIR / args.train_file)
    dev_lines = read_dev_lines(ROOT_DIR / args.dev_file)
    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"train_lines={len(train_lines)} dev_lines={len(dev_lines)} vocab={args.vocab}")

    rows = []
    original = bpe_tokenizer.EXTRA_CROSS_WORD_BIGRAMS
    try:
        for k in args.ks:
            bpe_tokenizer.EXTRA_CROSS_WORD_BIGRAMS = k
            tok = BPETokenizer(vocab_size=args.vocab)
            tok.train(train_lines)
            pkl = out_dir / f"tok1_mix_v{args.vocab}_k{k}.pkl"
            tok.save(str(pkl))
            sha = sha256_of(pkl)
            control = ""
            if k == 0:
                control = "MATCH" if sha == SHIPPED_TOK1_SHA else "MISMATCH"
            reloaded = BaseTokenizer.load(str(pkl))
            m = measure(reloaded, dev_lines)
            m.update({"k": k, "sha256": sha, "control": control, "pkl": str(pkl.relative_to(ROOT_DIR))})
            rows.append(m)
            print(
                f"k={k:<3} vocab={m['vocab_size']} tok/char={m['tokens_per_char']:.6f} "
                f"tok/sent={m['tokens_per_sentence']:.4f} bigrams={m['bigram_token_count']} "
                f"trigrams={m['trigram_token_count']} unk={m['unk_count']} "
                f"recon={m['reconstruction_rate']:.4f} {control} {sha[:12]}"
            )
    finally:
        bpe_tokenizer.EXTRA_CROSS_WORD_BIGRAMS = original

    base = next((r for r in rows if r["k"] == 0), None)
    if base:
        for r in rows:
            d_tps = (base["tokens_per_sentence"] - r["tokens_per_sentence"]) / base["tokens_per_sentence"] * 100
            print(f"  k={r['k']:<3} tok/sent {r['tokens_per_sentence']:.4f}  ({d_tps:+.2f}% vs k=0)")

    csv_path = ROOT_DIR / args.results_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["k", "vocab_size", "tokens_per_char", "tokens_per_sentence", "dev_tokens",
              "dev_sentences", "bigram_token_count", "trigram_token_count", "unk_count",
              "reconstruction_rate", "control", "sha256", "pkl"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
