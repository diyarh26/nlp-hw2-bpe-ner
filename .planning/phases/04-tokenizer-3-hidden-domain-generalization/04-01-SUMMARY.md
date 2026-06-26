---
phase: 04-tokenizer-3-hidden-domain-generalization
plan: 01
subsystem: tokenizer-data
tags: [mix, dedup, determinism, reproducibility, GEN-01]
provides:
  - Deterministic balanced tok3 training mix (D-01/D-02)
  - code/make_tok3_mix.py helper + manifest
requirements-completed:
  - GEN-01 (mix portion)
completed: 2026-06-26
status: complete
---

# Phase 4 Plan 1: Tok3 Mix + Manifest Summary

## What was built

- `code/make_tok3_mix.py` — standard-library-only CLI (`--seed --domain1 --domain2
  --output --manifest`). Implements D-01 (all of domain 2 + deterministic
  same-count SHA-256 sample of domain 1) and D-02 (exact-line dedup ONLY, after
  mixing, first-seen order; no lowercase/normalize/near-dedup).
- `data/tok3_mix_d1sample_d2all_seed42.txt` — the mix (~50 MB, gitignored under
  `data/`, regenerable from the helper).
- `tuning/results/tok3_mix_manifest.json` — seed, input/output SHA-256, line
  counts, sample count, dup-removed count.

## Result

- Sample count = domain-2 total = **319,696**; mixed = 319,696 + 319,696 =
  639,392 pre-dedup → **634,359** post-dedup (**5,033** exact duplicates removed).
- `output_sha256 = 82379e6a2fbbfe5e2e4c6204a95765140117dfaf828052e9c469bd53725ed804`.
- Balance check: domain-1 ~74 B/line, domain-2 ~92 B/line, so the 1:1-by-line
  sample is ~45% Twitter / ~55% news by bytes — balanced, slight news lean.

## Verification

- **Byte-reproducible:** a second run to a temp path produced an identical
  `output_sha256` and matching counts (manifest comparison `REPRODUCIBLE: True`).
  Proof retained as `tuning/results/tok3_mix_repro_check_manifest.json`; the
  oversized repro `.txt` copy was deleted (hash equality is the proof).
- `domain_1_sample_lines == domain_2_lines_total` (319,696).
- No transform/normalize logic in the helper (exact-line dedup only).
- `train_commands.txt` records the exact mix command; `trained_tokenizers/`
  untouched.

## Next

04-02: generate tok3 candidates at vocab 4000/8000/16000 via the unedited
`generate_tokenizers.py --train_files_3 tok3_mix_d1sample_d2all_seed42.txt`, then
measure CPU/OOV evidence and keep local gates green before VM spend.
