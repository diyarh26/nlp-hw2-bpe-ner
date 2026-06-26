---
phase: 03-per-domain-tuning
created: 2026-06-26
source: manual-codex-draft
depends_on:
  - 02-fast-encode-training-efficiency
requirements:
  - TUNE-01
  - TUNE-02
  - TUNE-03
  - TUNE-04
  - EVAL-03
---

# Phase 3 Context: Per-Domain Tuning

## Phase Goal

Tune tokenizer 1 for the Twitter/social domain and tokenizer 2 for the news domain by sweeping a small vocab-size grid, measuring CPU-local efficiency/speed, dispatching only the NER F1 leg to the Tesla M60 VM, and selecting the F1/efficiency knee without regressing Phase 1 gates.

## Design Decisions

- **Vocab grid:** `[4000, 8000, 16000]` for both tokenizer 1 and tokenizer 2.
  - Rationale: three points are enough to observe the small/medium/large vocab curve while keeping VM F1 runs bounded.
  - Candidate count: 2 domains x 3 vocab sizes = 6 candidate tokenizers and 6 VM NER runs.
- **Harness file:** create `code/tune_tokenizers.py`.
  - Reuse functions from `code/test_tokenizer.py` for tokens/char, speed, and reconstruction.
  - Train deterministic candidates with `BPETokenizer(vocab_size=...)` directly, not by editing `generate_tokenizers.py`.
  - Output candidate pkls under `tuning/tokenizers/`.
  - Output CPU metrics to `tuning/results/cpu_sweep.csv`.
- **CPU-local metrics:** measure tokens/char efficiency and tokens/sec speed on:
  - tokenizer 1 candidates: `data/domain_1_dev.txt`
  - tokenizer 2 candidates: `data/domain_2_dev.txt`
- **VM-only F1:** do not run NER locally during Phase 3 execution.
  - Create a VM runbook with exact `code/train_ner_model.py` commands for all six candidates.
  - Capture best F1 from each run into `tuning/results/vm_f1_results.csv`.
  - Assumption for budget: <=45 minutes per 20-epoch BiLSTM run on Tesla M60, so 6 runs <=4.5 hours plus setup/logging overhead, under the 10-hour budget.
  - Stop rule: if the first two VM runs average >60 minutes each, pause and shrink/confirm the grid before spending more VM time.
- **Selection rule:** for each domain, choose the smallest-vocab candidate within `0.02` absolute F1 of that domain's best F1; tie-break by lower tokens/char, then higher tokens/sec.
  - Rationale: protects F1 first, then selects the efficiency knee rather than blindly maximizing vocab.
- **Submission/regression:** after selection, stage selected tokenizer 1/2 plus the current Phase-2 tokenizer 3 baseline, rebuild `HW2_214034621.zip`, and rerun Phase 1/2 gates.

## CPU vs VM Boundary

CPU-local:

- Train candidate tokenizer pkls.
- Measure tokens/char efficiency.
- Measure tokenization speed.
- Run `speed_check.py --only all`, `align_check.py`, fresh loads, zip/checker once candidates are selected.

VM-only:

- Run `code/train_ner_model.py` for candidate F1.
- Record best F1 per candidate from fixed 20-epoch NER training.
- Do not spend VM time until all CPU candidate generation/metrics are complete and local gates are green.

## Open For Confirmation

- Confirm the vocab grid `[4000, 8000, 16000]`. If the VM budget feels tighter than assumed, use `[4000, 8000]` first and add `16000` only if needed.
- Confirm the VM runtime budget assumption: <=45 minutes per candidate run, <=4.5 hours for six F1 runs plus overhead.
- Confirm the knee rule: choose smallest vocab within `0.02` absolute F1 of best, then lower tokens/char, then higher tokens/sec.
- Confirm whether Phase 3 should update `train_commands.txt` immediately with selected tok1/tok2 commands, or defer final command freeze to Phase 5 (`SERDE-04`).

