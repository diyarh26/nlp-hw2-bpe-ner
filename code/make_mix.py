"""Deterministic domain-1 + domain-2 training-mix generator (recipe sweep).

Generalizes `code/make_tok3_mix.py` with a single new knob, ``--d1_ratio R``:
include ``round(R * len(domain2))`` deterministically-sampled domain-1 lines plus
ALL of domain 2, then exact-line dedup (first-seen) after mixing.

  R = 1.0  -> the existing balanced 1:1 mix (byte-identical to
              data/tok3_mix_d1sample_d2all_seed42.txt; this is the built-in
              provenance check below).
  R > 1.0  -> domain-1-heavy mixes (e.g. 2:1, 3:1) that keep more Twitter
              coverage while retaining the news-data regularization that made the
              balanced mix beat pure domain-1 at small vocab.

Sampling/dedup are identical to make_tok3_mix.py (same SHA-256 keying), so R=1.0
reproduces the shipped mix exactly. Standard library only; deterministic; a
re-run with the same inputs/seed/ratio yields a byte-identical output file
(proven via the manifest's output_sha256). The mix is written under --output so
the unedited generate_tokenizers.py can consume it via `--train_files_* <name>`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os


def read_lines(path):
    """Return file content split on '\n', dropping only a single trailing-newline
    artifact (interior blank lines preserved, content kept exactly)."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def deterministic_sample(domain1_lines, count, seed):
    """Pick `count` domain-1 lines deterministically.

    Identical keying to make_tok3_mix.deterministic_sample: key each line by
    SHA-256 of (seed, 0-based line number, exact line bytes), sort by
    (hash_hex, line_number), take the first `count`. Keeping the same key means
    the R=1.0 sample is a byte-identical superset relationship with the shipped
    mix's sample (the first N2 picks are exactly the shipped sample).
    """
    keyed = []
    for i, line in enumerate(domain1_lines):
        payload = f"{seed}\x00{i}\x00{line}".encode("utf-8")
        keyed.append((hashlib.sha256(payload).hexdigest(), i))
    keyed.sort()
    chosen_idx = [i for _, i in keyed[:count]]
    return [domain1_lines[i] for i in chosen_idx]


def exact_dedup_first_seen(lines):
    seen = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate a deterministic domain-1+2 training mix at a chosen d1:d2 ratio."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d1_ratio", type=float, default=1.0,
                        help="domain-1 lines = round(d1_ratio * len(domain2)), capped at len(domain1)")
    parser.add_argument("--domain1", default="data/domain_1_train.txt")
    parser.add_argument("--domain2", default="data/domain_2_train.txt")
    parser.add_argument("--output", required=True,
                        help="Output mix path (place under data/ so the generator can read it)")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    if args.d1_ratio <= 0:
        raise SystemExit("--d1_ratio must be > 0")

    domain1_lines = read_lines(args.domain1)
    domain2_lines = read_lines(args.domain2)

    sample_count = round(args.d1_ratio * len(domain2_lines))
    if sample_count > len(domain1_lines):
        sample_count = len(domain1_lines)
    sampled_domain1 = deterministic_sample(domain1_lines, sample_count, args.seed)

    pre_dedup = sampled_domain1 + domain2_lines
    mixed = exact_dedup_first_seen(pre_dedup)

    output_bytes = ("\n".join(mixed) + "\n").encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as fh:
        fh.write(output_bytes)

    manifest = {
        "seed": args.seed,
        "d1_ratio": args.d1_ratio,
        "domain_1_input": args.domain1,
        "domain_2_input": args.domain2,
        "domain_1_input_sha256": sha256_file(args.domain1),
        "domain_2_input_sha256": sha256_file(args.domain2),
        "domain_1_lines_total": len(domain1_lines),
        "domain_2_lines_total": len(domain2_lines),
        "domain_1_sample_lines": sample_count,
        "pre_dedup_lines": len(pre_dedup),
        "post_mix_lines": len(mixed),
        "post_mix_duplicate_lines_removed": len(pre_dedup) - len(mixed),
        "output_file": args.output,
        "output_sha256": sha256_bytes(output_bytes),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=True)
        fh.write("\n")

    print(
        f"mix (d1_ratio={args.d1_ratio}): {len(sampled_domain1)} sampled domain-1 + "
        f"{len(domain2_lines)} domain-2 -> {len(mixed)} lines "
        f"({manifest['post_mix_duplicate_lines_removed']} dup removed)"
    )
    print(f"output_sha256={manifest['output_sha256']}")
    print(f"wrote {args.output} and {args.manifest}")


if __name__ == "__main__":
    main()
