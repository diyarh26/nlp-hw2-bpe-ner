---
phase: 04-tokenizer-3-hidden-domain-generalization
plan: 04
subsystem: tokenizer
tags: [selection, staging, hidden-domain, reproducibility, submission]
provides:
  - Selected + staged tok3 vocab 8000 (mix-trained)
  - code/select_tok3.py hidden-generalization scorer
  - tuning/results/tok3_selection.csv
requirements-completed:
  - GEN-01
  - GEN-02
  - GEN-03
completed: 2026-06-27
status: complete
---

# Phase 4 Plan 4: Tok3 Selection + Staging Summary

## Decision: tok3 = vocab 8000 (mix-trained)

`code/select_tok3.py` implements the approved hidden-generalization score
(0.35 mean visible F1 + 0.25 tokens/char + 0.15 domain-balance + 0.10 OOV +
0.10 reconstruction gate + 0.05 speed). Auditable result in
`tuning/results/tok3_selection.csv`:

| vocab | score | mean F1 | Twitter F1 | news F1 | comb tok/char |
|------:|------:|--------:|-----------:|--------:|--------------:|
| 4000  | 0.7500 | 0.7033 | 0.4450 | 0.9616 | 0.3266 |
| 8000  | 0.6759 | 0.6902 | 0.4217 | 0.9587 | 0.2879 |
| 16000 | 0.4500 | 0.6688 | 0.3792 | 0.9585 | 0.2602 |

Key finding: tok3 F1 **decreases monotonically with vocab on both visible
domains** — the mix-trained tokenizer over-merges at larger vocab. So 16000 is
rejected (worst F1 and lowest score).

**The score's top pick is 4000**, but the **user selected 8000** as a deliberate
override: efficiency is an equally-weighted competition axis and 4000 is the
efficiency-worst (+14% tokens vs 8000, +26% vs 16000) for only ~0.013 mean
visible-F1 (partly Twitter-NER noise). 8000 is the balanced knee — near-best
generalization F1, materially better efficiency, in the recommended medium-vocab
range, and not worst on any axis (safest profile for an unknown domain). The
score CSV is retained as honest evidence; this override is the staged choice.

## Staging + verification

- Copied `tuning/tokenizers/tokenizer_3_vocab8000.pkl` ->
  `trained_tokenizers/tokenizer_3.pkl` (old Phase-2 baseline `ef285be3...`
  replaced by mix-trained `f03aac21...`).
- **Reproducible:** staged tok3 is byte-identical (`f03aac21...`) to a fresh
  unedited `generate_tokenizers.py --vocab_size 8000 --seed 42 --train_files_3
  tok3_mix_d1sample_d2all_seed42.txt` run.
- tok1 (`8833951871fa3b3f...`) and tok2 (`ae7c16f487104325...`) SHA-256
  **unchanged**; vocab sizes 8000 / 16000 / 8000.
- Gates green: `align_check` PASS, `speed_check --only all` PASS (cache + train),
  `build_submission` rebuilt `HW2_214034621.zip`, `check_submission` valid
  (tok3 bigram `,"▁the`, no `[ERROR]`), forbidden-file hashes intact.
- `train_commands.txt` updated with the final tok3 recipe (mix + generator +
  copy); the stale Phase-2 tok3 baseline line marked superseded.

## Next

Phase 4 closes GEN-01/02/03. Phase 5: final all-tokenizer VM F1 + competition
measurement (reuse the tok3 visible-domain VM rows already captured in 04-03).
The 3 final tokenizers are byte-identical to candidates already on the VM
(`tokenizer_{1_vocab8000,2_vocab16000,3_vocab8000}.pkl`), so Phase 5 needs no
new candidate push.
