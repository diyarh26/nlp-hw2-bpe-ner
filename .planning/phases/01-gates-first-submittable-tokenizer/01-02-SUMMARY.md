---
phase: 01-gates-first-submittable-tokenizer
plan: 02
subsystem: tokenizer
tags: [bpe, tokenizer, alignment, byte-fallback, ner]
requires:
  - phase: 01-01
    provides: Scaffold, data, uv environment, red align_check harness
provides:
  - Working BPETokenizer.train
  - Working BPETokenizer.encode
  - Working BPETokenizer.decode
  - VP-1/2/4/6 green local harness
affects:
  - 01-03
  - 01-04
tech-stack:
  added: []
  patterns:
    - Deterministic char-up BPE with lexicographic tie-break
    - Mergeable U+2581 space marker capped at one internal space
    - Private-use byte-token fallback for zero UNK and exact UTF-8 decode
key-files:
  created: []
  modified:
    - code/bpe_tokenizer.py
key-decisions:
  - "Represent fallback bytes as private-use-prefixed tokens so decode can reconstruct unseen UTF-8 characters exactly."
  - "Use a simple full-recount merge loop in Phase 1; Phase 2 owns training/encode speed optimization."
  - "Keep TRAIN_TEXT_LIMIT at 5000 for the Phase-1 simple trainer so local generation remains practical."
patterns-established:
  - "All legal merges satisfy `(left + right).count(space_token) <= 1` in both train and encode."
  - "The local `align_check.py` default run is already green for VP-1, VP-2, VP-3, VP-4, and VP-6."
requirements-completed:
  - CORE-01
  - CORE-02
  - CORE-03
  - CORE-04
  - CORE-05
  - ALIGN-01
  - ALIGN-02
  - ALIGN-03
duration: 7 min
completed: 2026-06-25
status: complete
---

# Phase 1 Plan 2: Char-Up BPE Tokenizer Summary

**A simple correct BPE tokenizer now trains, encodes, decodes, guarantees a legal bigram, avoids trigram tokens, and passes the local alignment harness.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-06-25T11:48:00Z
- **Completed:** 2026-06-25T11:55:00Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- Implemented `BPETokenizer.train()` with deterministic char-up BPE, sorted base vocab assignment, lexicographic tie-breaking, and a first-N training cap.
- Added the required `space_token = "▁"` as a first-class mergeable symbol.
- Enforced the one-internal-space cap for both training and encode-time merges.
- Guaranteed at least one cross-word bigram by force-merging the highest-count legal cross-space pair when needed.
- Implemented `encode()` with UTF-8 byte fallback so `[UNK]` id 1 is never emitted.
- Implemented `decode()` with exact reconstruction for byte-fallback characters and spaces.

## Task Commits

1. **Task 1: Implement train()** - `7c28e47` (`feat`)
2. **Task 2: Implement encode() and decode()** - `7c28e47` (`feat`)
3. **Task 3: Wire align_check.py gates** - already covered by existing harness; no harness weakening was required

**Plan metadata:** pending in close-out commit.

## Files Created/Modified

- `code/bpe_tokenizer.py` - Replaced the course stub with a working Phase-1 tokenizer implementation.

## Verification

- Train probe: PASS - `space_token == "▁"`, `space_token` in vocab, at least one bigram, no trigram+ token.
- Encode/decode probe: PASS - ASCII, double-space, non-ASCII, and emoji samples reconstruct without `[UNK]` and with monotonic cumulative decode lengths.
- `uv run python code/align_check.py --only vp1,vp2,vp4,vp6`: PASS.
- `uv run python code/align_check.py`: PASS for VP-1, VP-2, VP-3, VP-4, and VP-6.
- Byte fallback alphabet check: PASS - all 256 byte tokens are in the vocab after training.
- Special decode check: PASS - PAD/UNK/BOS/EOS decode to length-0 surface.
- `py_compile` for `code/bpe_tokenizer.py` and `code/align_check.py`: PASS.
- Forbidden-file hash recheck: PASS - the four forbidden files still match the baseline.

## Decisions Made

- Chose private-use-prefixed byte token strings instead of latin-1 raw byte characters. This avoids collisions with real corpus characters while still allowing exact UTF-8 reconstruction in `decode()`.
- Did not add `__getstate__`/`__setstate__` here. The current state is already plain enough for VP-3, but 01-03 remains the hardening/reproducibility plan.
- Kept the trainer deliberately simple; speed improvements remain Phase 2 work.

## Deviations from Plan

### Auto-fixed Issues

None - plan executed within scope. The only adjustment was using private-use byte tokens for a safer exact byte fallback representation.

---

**Total deviations:** 0 auto-fixed.
**Impact on plan:** No scope creep; all required local gates for 01-02 are green.

## Issues Encountered

- One verification probe failed while printing the `▁` marker through the Windows console encoding; rerunning with escaped output passed. This did not indicate a tokenizer failure.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `01-03`: add explicit serialization hardening (`__getstate__`/`__setstate__`, regex/cache handling if needed) and prove byte-reproducibility across two unedited `generate_tokenizers.py` runs.

## Self-Check: PASSED

- `code/bpe_tokenizer.py` exists and implements train/encode/decode.
- `code/align_check.py` is green for the 01-02 local gates.
- Forbidden files were not modified.

---
*Phase: 01-gates-first-submittable-tokenizer*
*Completed: 2026-06-25*
