---
phase: 02-fast-encode-training-efficiency
plan: 03
subsystem: tokenizer
tags: [training, incremental-bpe, performance, reproducibility, submission]
requires:
  - phase: 02-02
    provides: Cached encode path + encode/train oracle harness
provides:
  - Uncapped weighted word-chunk BPE trainer (full-corpus capable)
  - Integer-symbol incremental merge loop with local pair-count deltas
  - Regenerated full-corpus tokenizer_{1,2,3}.pkl at vocab 8000
  - Checker-clean rebuilt HW2_214034621.zip
affects:
  - 03-01
tech-stack:
  added: []
  patterns:
    - Word-chunk frequency dedup (split at each space marker)
    - Integer-interned symbols with precomputed space counts
    - Local-neighbour pair-count deltas (no whole-chunk recount)
    - Lazy max-heap, push only changed pairs
    - Superset pair->chunks inverted index
key-files:
  created: []
  modified:
    - code/bpe_tokenizer.py
    - code/speed_check.py
    - train_commands.txt
requirements-completed:
  - SPEED-03
duration: ~ (cross-agent relay; trainer fix + generation)
completed: 2026-06-26
status: complete
---

# Phase 2 Plan 3: Weighted Incremental Trainer Summary

**The capped Phase-1 trainer is gone: BPE now trains on the full corpora via an integer-symbol, local-delta incremental merge loop, and the regenerated submission stays checker-clean and byte-reproducible.**

## What was delivered

- **Word-chunk training representation** (Codex WIP, retained): `_text_to_chunks` tiles `_text_to_train_symbols` and starts a new chunk at each `▁`; `_weighted_chunk_sequences` dedups identical chunks with integer weights. `FORCED_BIGRAM_SAMPLE_LINES = 4000` + `_forced_cross_word_bigram_pair` supply the cross-word bigram that chunk-only training cannot produce on its own.
- **Efficient incremental merge loop** (`_train_merges`, Claude): the merge bookkeeping was rewritten because chunk dedup alone still timed out (domain 1 has 1,212,080 unique chunks; 72.7% are singletons). The new loop:
  - interns symbols to dense integer ids with precomputed space-marker counts (legality is an int add, not a string scan);
  - applies each merge with **local-neighbour pair-count deltas** at each merge site instead of recomputing every pair in the chunk;
  - keeps a **lazy max-heap**, pushing only pairs whose count changed (no per-occurrence heap churn);
  - maintains `pair_chunks` as a **superset** inverted index (entries only added, never pruned on decrement) so a merge never misses a chunk that still holds the pair.
  - Removed the old `_build_pair_index` / `_apply_indexed_merge` / `_add`/`_remove_sequence_pairs` / `_best_indexed_pair` / `_push_pair_heap` helpers.
- **Identical-result guarantee:** the new loop produces merges/vocab byte-identical to the brute-force `train_full_recount_oracle`, so every downstream gate is unaffected.
- **Regenerated artifacts:** all three `tokenizer_{1,2,3}.pkl` at `--vocab_size 8000`, rebuilt `HW2_214034621.zip`, `train_commands.txt` annotated with the uncapped-trainer method.

## Performance

- Domain-1 (95 MB, 1.28 M lines) train: **vocab 1000 → 73 s**, **vocab 8000 → 124 s** (previously timed out at 5 min for vocab 1000).
- Full 3-tokenizer generation (`generate_tokenizers.py --vocab_size 8000 --seed 42`): **273 s** (~4.5 min) — previously aborted/uncompletable.

## Verification

- `py -m uv run python code\speed_check.py --only all`: PASS — `ORACLE: PASS (83)`, `CACHE: PASS`, `TRAIN: PASS - incremental trainer matched full recount for 105 merges`.
- `py -m uv run python code\align_check.py`: PASS — VP1, VP2, VP3, VP4, VP6.
- Full generation: all three pkls written and load via `BaseTokenizer.load` at vocab 8000.
- `check_submission.py HW2_214034621.zip` (PYTHONIOENCODING=utf-8): **exit 0, no `[ERROR]`**; bigrams `I▁miss` (tok1), `,"▁the` (tok2), `an▁Me` (tok3).
- Byte-reproducibility: two unedited `generate_tokenizers.py --vocab_size 600 --seed 42` runs → SHA-256-identical `tokenizer_{1,2,3}.pkl`.
- Forbidden-file SHA-256 recheck vs `01-.../forbidden_files.sha256`: all four match (only `code/bpe_tokenizer.py` was modified).

## Deviations from Plan

- The plan's Task 1 trainer (committed `5d4945e`) and a follow-up chunk-dedup pass were **functionally correct but not fast enough** for full-corpus generation. The merge-loop internals were rewritten (int symbols + local deltas + disciplined heap) to make uncapped generation feasible. No training-data cap was reintroduced (prohibition respected).

## Cross-agent note

This plan spanned a Claude⇄Codex relay. Codex implemented the chunk-level training and committed it; the feasibility checkpoint failed twice (full generation still timed out). Claude profiled the bottleneck (1.2 M unique chunks; heap churn + whole-chunk recount), rewrote `_train_merges`, and completed generation + submission + close-out.

## Next Phase Readiness

Phase 2 complete (SPEED-01/02/03). Ready for **Phase 3: Per-Domain Tuning (Tokenizers 1 & 2)** — vocab-size sweep scored on tokens/char efficiency, tokens/sec speed, and NER F1. The F1 leg of the sweep is the first work that needs the Tesla M60 VM.

## Self-Check: PASSED

- Full corpora train with no `TRAIN_TEXT_LIMIT`.
- Incremental trainer matches the full-recount oracle (identical merges/vocab).
- Submission rebuilt, checker-clean, byte-reproducible, forbidden files intact.

---
*Phase: 02-fast-encode-training-efficiency*
*Completed: 2026-06-26*
