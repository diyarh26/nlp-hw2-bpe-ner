---
phase: 03-per-domain-tuning
plan: 03
subsystem: tokenizer
tags: [selection, staging, reproducibility, submission]
provides:
  - Selected tok1 vocab 8000
  - Selected tok2 vocab 16000
  - Staged trained_tokenizers/tokenizer_1.pkl and tokenizer_2.pkl
requirements-completed:
  - TUNE-04
completed: 2026-06-26
status: complete
---

# Phase 3 Plan 3: Selection and Staging Summary

## Decision

Selected:

- `tokenizer_1.pkl`: vocab `8000`
- `tokenizer_2.pkl`: vocab `16000`
- `tokenizer_3.pkl`: kept as the Phase 2 vocab-8000 baseline for Phase 4

## Evidence

Tok1 selection is direct: vocab 8000 had the best VM F1 (`0.4352`) while also improving token efficiency over vocab 4000. Vocab 16000 was more efficient, but its F1 fell to `0.3692`, outside the approved knee band.

Tok2 selection follows the user-approved competition tradeoff: vocab 16000 had the best token efficiency (`0.22937417` tokens/char) and fastest VM wall time (`3.2012` minutes), while its F1 (`0.9557`) was only `0.0053` below the best tok2 F1 (`0.9610` at vocab 4000). This keeps the candidate inside the `0.02` F1 band and optimizes efficiency/speed.

## Staging

- Copied `tuning/tokenizers/tokenizer_1_vocab8000.pkl` to `trained_tokenizers/tokenizer_1.pkl`.
- Copied `tuning/tokenizers/tokenizer_2_vocab16000.pkl` to `trained_tokenizers/tokenizer_2.pkl`.
- Preserved `trained_tokenizers/tokenizer_3.pkl` as the Phase 2 vocab-8000 baseline.
- Updated `train_commands.txt` with exact reproduction commands for staged tok1/tok2 and current tok3.

## Verification

- Staged tok1 matched the measured candidate SHA-256:
  `8833951871FA3B3F167094F80B34F511C34E1E5787B6F5481B4CB05B3C57AC3D`
- Staged tok2 matched the measured candidate SHA-256:
  `AE7C16F4871043256FD115B27CA399FFC99E5FA564A4FAE7230C34DA2BB52177`
- Staged tok3 matched the Phase 2 vocab-8000 generator output SHA-256:
  `EF285BE362DE660A19C96344B69E2D32446B5220596F039604ED3DF6507A4AA9`
- Fresh unedited generator comparison passed:
  - `trained_tokenizers/tokenizer_1.pkl` byte-identical to `tuning/_cmp/8000/tokenizer_1.pkl`
  - `trained_tokenizers/tokenizer_2.pkl` byte-identical to `tuning/_cmp/16000/tokenizer_2.pkl`
- All three staged tokenizers load via `BaseTokenizer.load`.

## Next Phase Readiness

Phase 3 is complete. Phase 4 should replace the Phase 2 tok3 baseline with a hidden-domain generalization tokenizer trained on a deliberate domain 1+2 mix.

