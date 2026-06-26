"""Deterministic tokenizer-3 training-mix generator (Phase 4, GEN-01).

Implements the locked Phase 4 decisions:
  D-01  Balanced mix: ALL of domain 2 plus a deterministic same-count sample of
        the larger domain 1, so tok3 does not become Twitter-heavy.
  D-02  Exact-line dedup ONLY, applied after mixing, first-seen order. No
        lowercasing, Unicode normalization, punctuation stripping, or near-dedup
        (casing/punctuation may carry NER signal).

Standard library only. Deterministic: a second run with the same inputs/seed
produces a byte-identical output file (proven via the manifest's output_sha256).
The mix is written under --data_dir's folder so the unedited generate_tokenizers.py
can consume it via `--train_files_3 <basename>`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os


def read_lines(path):
    """Return file content as a list of lines split on '\n'.

    A single trailing-newline artifact (empty final element) is dropped, but
    genuine interior blank lines are preserved. Line content is kept exactly as
    decoded (no strip/normalize) so dedup is exact per D-02.
    """
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

    Key each line by SHA-256 of (seed, 0-based line number, exact line bytes),
    sort by (hash_hex, line_number), and take the first `count`. The line number
    in the key keeps the order total and stable even for repeated identical
    lines (which are only collapsed later by the required post-mix exact dedup).
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
        description="Generate the deterministic balanced+deduped tok3 training mix."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain1", default="data/domain_1_train.txt")
    parser.add_argument("--domain2", default="data/domain_2_train.txt")
    parser.add_argument(
        "--output", default="data/tok3_mix_d1sample_d2all_seed42.txt"
    )
    parser.add_argument(
        "--manifest", default="tuning/results/tok3_mix_manifest.json"
    )
    args = parser.parse_args()

    domain1_lines = read_lines(args.domain1)
    domain2_lines = read_lines(args.domain2)

    sample_count = len(domain2_lines)  # D-01: same line count as all of domain 2
    sampled_domain1 = deterministic_sample(domain1_lines, sample_count, args.seed)

    # D-02: concatenate sampled domain 1 + all domain 2, then exact-line dedup.
    pre_dedup = sampled_domain1 + domain2_lines
    mixed = exact_dedup_first_seen(pre_dedup)

    output_bytes = ("\n".join(mixed) + "\n").encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as fh:
        fh.write(output_bytes)

    manifest = {
        "seed": args.seed,
        "domain_1_input": args.domain1,
        "domain_2_input": args.domain2,
        "domain_1_input_sha256": sha256_file(args.domain1),
        "domain_2_input_sha256": sha256_file(args.domain2),
        "domain_1_lines_total": len(domain1_lines),
        "domain_2_lines_total": len(domain2_lines),
        "domain_1_sample_lines": sample_count,
        "domain_1_unique_lines": len(set(domain1_lines)),
        "domain_2_unique_lines": len(set(domain2_lines)),
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
        f"tok3 mix: {len(sampled_domain1)} sampled domain-1 + "
        f"{len(domain2_lines)} domain-2 -> {len(mixed)} lines "
        f"({manifest['post_mix_duplicate_lines_removed']} dup removed)"
    )
    print(f"output_sha256={manifest['output_sha256']}")
    print(f"wrote {args.output} and {args.manifest}")


if __name__ == "__main__":
    main()
