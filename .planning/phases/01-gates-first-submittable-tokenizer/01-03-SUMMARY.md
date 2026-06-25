---
phase: 01-gates-first-submittable-tokenizer
plan: 03
subsystem: tokenizer
tags: [pickle, reproducibility, serialization, bpe]
requires:
  - phase: 01-02
    provides: Working BPETokenizer train/encode/decode
provides:
  - Explicit BPETokenizer pickle state protocol
  - Rebuilt runtime regex/cache helpers after load
  - Byte-reproducibility proof for tokenizer_{1,2,3}.pkl
affects:
  - 01-04
tech-stack:
  added: []
  patterns:
    - Primitive-only __getstate__/__setstate__
    - Runtime helpers rebuilt from persisted strings
    - Generator double-run hash comparison
key-files:
  created: []
  modified:
    - code/bpe_tokenizer.py
key-decisions:
  - "Keep learned tokenizer state primitive-only and exclude compiled regex/cache objects from pickle."
  - "Preserve vocab_size as a hard cap when forcing the required cross-word bigram."
  - "Verify reproducibility across all three tokenizer outputs, not only tokenizer_1.pkl."
patterns-established:
  - "Use `py -m uv run python generate_tokenizers.py --vocab_size 600 --output_dir <dir>` for fast reproducibility checks."
  - "Hash-compare generated pickles and remove temp output directories after verification."
requirements-completed:
  - SERDE-01
  - SERDE-02
  - SERDE-03
duration: 14 min
completed: 2026-06-25
status: complete
---

# Phase 1 Plan 3: Serialization and Reproducibility Summary

**BPETokenizer now has an explicit primitive-only pickle contract and all three generated tokenizer pickles are byte-reproducible across repeated unedited generator runs.**

## Performance

- **Duration:** 14 min
- **Started:** 2026-06-25T11:55:00Z
- **Completed:** 2026-06-25T12:09:00Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- Added `BPETokenizer.__getstate__()` to persist only primitive tokenizer state.
- Added `BPETokenizer.__setstate__()` to restore state and rebuild runtime helpers.
- Added a stored `pretoken_pattern` string, a compiled runtime regex helper, and an empty runtime encode cache.
- Fixed the forced-bigram fallback so it does not exceed `vocab_size`.
- Proved all three generated tokenizer pickles are byte-identical across two unedited `generate_tokenizers.py` runs.

## Task Commits

1. **Task 1: Add explicit pickle state protocol** - `e57313e` (`fix`)
2. **Task 2: Prove deterministic generation** - verification-only; no tracked files
3. **Task 3: Confirm full local gate** - verification-only; no tracked files

**Plan metadata:** pending in close-out commit.

## Files Created/Modified

- `code/bpe_tokenizer.py` - Added explicit serialization hooks, runtime helper rebuilding, and the vocab-cap fix for forced bigrams.

## Verification

- In-process pickle probe: PASS - `pickle.loads(pickle.dumps(t))` round-trips encode/decode and pickled bytes do not include compiled regex/cache state.
- `py -m uv run python code/align_check.py --only vp3`: PASS.
- `py -m uv run python code/align_check.py`: PASS for VP-1, VP-2, VP-3, VP-4, VP-6.
- Reproducibility double-run at `--vocab_size 600`: PASS for all three generated pickles:
  - `tokenizer_1.pkl`: `D452CF909CF98D9E0F9D94753EA82551C5105B8B5F67510D153D0AB0F37F087C`
  - `tokenizer_2.pkl`: `A13F3E9C96AC2612D2549DE7B9D326068E0852B0F4FC2B8A663311F7C042B722`
  - `tokenizer_3.pkl`: `D452CF909CF98D9E0F9D94753EA82551C5105B8B5F67510D153D0AB0F37F087C`
- Generated vocab-size check: PASS - all three generated tokenizers stayed `<= 600`.
- `py_compile` for `code/bpe_tokenizer.py` and `code/align_check.py`: PASS.
- Forbidden-file hash recheck: PASS.

## Decisions Made

- Kept regex as a stored pattern string (`PRETOKEN_PATTERN`) and compiled it only into runtime state. The current tokenizer does not need regex for behavior yet, but this satisfies the planned serialization contract and leaves a safe hook for later pre-tokenization work.
- Treated the `vocab_size` cap as strict. The forced-bigram path now uses the last available vocab slot rather than adding a token after the cap.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Forced bigram could exceed vocab_size**
- **Found during:** Task 2 reproducibility run
- **Issue:** Tokenizer 2 produced vocab size 601 for `--vocab_size 600` because the forced-bigram fallback added after the loop reached the cap.
- **Fix:** Reserve the final vocab slot for a forced legal bigram if no bigram exists yet, and only force after the loop when there is remaining capacity.
- **Files modified:** `code/bpe_tokenizer.py`
- **Verification:** Generator double-run now reports all three generated tokenizers at vocab size 600 for `--vocab_size 600`.
- **Committed in:** `e57313e`

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Improved correctness without changing scope; reproducibility and hard gates remain green.

## Issues Encountered

None remaining.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `01-04`: generate the real Phase-1 tokenizers, create `train_commands.txt`, run the local CPU NER smoke, build `HW2_214034621.zip`, and run `check_submission.py`.

## Self-Check: PASSED

- Fresh-process pickle reload passes.
- All local alignment gates pass.
- All three generated pickles are reproducible across repeated unedited generator runs.
- Forbidden files were not modified.

---
*Phase: 01-gates-first-submittable-tokenizer*
*Completed: 2026-06-25*
