---
phase: 03-per-domain-tuning
plan: 01
subsystem: tokenizer
tags: [tuning, cpu-sweep, measurement, reproducibility]
provides:
  - Six generator-produced tok1/tok2 candidate pickles
  - Measure-only CPU sweep harness
  - CPU efficiency/speed table for vocab 4000/8000/16000
requirements-completed:
  - TUNE-01
  - TUNE-02
completed: 2026-06-26
status: complete
---

# Phase 3 Plan 1: CPU Candidate Sweep Summary

## What was delivered

- Generated candidate tokenizers with the unedited `generate_tokenizers.py` for vocab sizes `4000`, `8000`, and `16000`.
- Copied only tok1/tok2 candidates into `tuning/tokenizers/`:
  - `tokenizer_1_vocab4000.pkl`, `tokenizer_1_vocab8000.pkl`, `tokenizer_1_vocab16000.pkl`
  - `tokenizer_2_vocab4000.pkl`, `tokenizer_2_vocab8000.pkl`, `tokenizer_2_vocab16000.pkl`
- Added `code/tune_tokenizers.py` as a measure-only harness. It loads candidate pickles, measures dev tokenization efficiency/speed, checks reconstruction samples, and writes `tuning/results/cpu_sweep.csv`.

## CPU Sweep Results

| tokenizer | vocab | tokens/char | tokens/sec | reconstruction |
|-----------|------:|------------:|-----------:|---------------:|
| tok1 | 4000 | 0.31217898 | 90666450.5186 | 1.000000 |
| tok1 | 8000 | 0.28061540 | 68892441.7324 | 1.000000 |
| tok1 | 16000 | 0.25758830 | 66164619.4114 | 1.000000 |
| tok2 | 4000 | 0.30712300 | 86905579.4882 | 1.000000 |
| tok2 | 8000 | 0.26098571 | 98587372.5857 | 1.000000 |
| tok2 | 16000 | 0.22937417 | 68898020.1726 | 1.000000 |

## Verification

- `tuning/results/cpu_sweep.csv` exists with exactly six rows.
- All six candidates load via `BaseTokenizer.load`.
- Reconstruction rate is `1.0` for every candidate.
- `trained_tokenizers/` was not overwritten during candidate generation.

