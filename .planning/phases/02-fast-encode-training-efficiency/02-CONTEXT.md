---
phase: 02-fast-encode-training-efficiency
created: 2026-06-25
source: inline-gsd-plan-phase
depends_on:
  - 01-gates-first-submittable-tokenizer
requirements:
  - SPEED-01
  - SPEED-02
  - SPEED-03
---

# Phase 2 Context: Fast Encode + Training Efficiency

## Phase Goal

Replace Phase 1's correctness-first slow paths with a faster merge-rank encoder, cache repeated encode work, and replace the temporary `TRAIN_TEXT_LIMIT = 1000` fallback with weighted/incremental BPE training that can train the real corpora in reasonable local wall time.

## Current Baseline

- `code/bpe_tokenizer.py` already persists `merge_ranks` and uses a rank lookup in `_apply_ranked_merges`, but the loop repeatedly scans and rebuilds the full symbol list and does not use `_encode_cache`.
- `TRAIN_TEXT_LIMIT = 1000` was introduced in `01-04` only to make Phase 1's fallback artifact feasible. It is not an acceptable final training strategy for the speed/training-efficiency axis.
- Phase 1 gates are green:
  - `code/align_check.py` passes.
  - Three `--vocab_size 8000` tokenizers can be generated at the capped baseline.
  - `HW2_214034621.zip` passes `check_submission.py` with no `[ERROR]` when `PYTHONIOENCODING=utf-8`.
  - Local CPU NER on domain 1 reached best dev F1 `0.4218`.

## Locked Constraints

- Do not edit:
  - `code/base_tokenizer.py`
  - `code/train_ner_model.py`
  - `generate_tokenizers.py`
  - `check_submission.py`
- Do not add dependencies beyond the locked stack.
- Preserve decode length-faithfulness and zero-UNK byte fallback.
- Preserve at least one legal cross-word bigram and no token spanning 3+ words.
- Preserve pickle compatibility: no compiled regex, cache contents, lambdas, closures, or local classes in persisted state.

## Phase 2 Strategy

1. Add a benchmark/oracle harness before changing behavior. It should compare the current trained tokenizer output with an explicit slow merge replay and measure repeated encode calls.
2. Rewrite encode as a cached, rank-map greedy path. The public `encode(text)` result must stay byte-identical to the oracle.
3. Rewrite training around weighted unique sequences, weighted pair counts, and an inverted `pair -> sequence ids` index. Remove reliance on `TRAIN_TEXT_LIMIT` for normal training while keeping determinism.

## Non-Goals

- No NER hyperparameter changes.
- No VM/GPU run in Phase 2.
- No vocab-size/domain tuning yet; Phase 3 owns tuning.
- No hidden-domain mix optimization yet; Phase 4 owns tokenizer 3 strategy.

