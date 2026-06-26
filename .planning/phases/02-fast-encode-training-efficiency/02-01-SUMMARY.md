---
phase: 02-fast-encode-training-efficiency
plan: 01
subsystem: tokenizer
tags: [oracle, benchmark, cache, speed]
requires: []
provides:
  - Encode slow-replay oracle harness
  - Cache/pickle probe harness
  - Baseline training probe hook
affects:
  - 02-02
  - 02-03
tech-stack:
  added: []
  patterns:
    - Framework-free harness script
    - Slow merge replay independent of public encode
    - Runtime cache pickle-drop probe
key-files:
  created:
    - code/speed_check.py
  modified: []
key-decisions:
  - "Use recorded `tokenizer.merges` replay as the behavior oracle for encode rewrites."
  - "Keep the cache speed check permissive in 02-01 because the real cached public encode lands in 02-02."
patterns-established:
  - "`py -m uv run python code\\speed_check.py --only oracle` for encode equality."
  - "`py -m uv run python code\\speed_check.py --only cache` for timing and pickle cache-drop checks."
requirements-completed: []
duration: 8 min
completed: 2026-06-25
status: complete
---

# Phase 2 Plan 1: Oracle and Benchmark Harness Summary

**Phase 2 now has a reusable speed/oracle harness before optimizer changes land.**

## Performance

- **Duration:** 8 min
- **Completed:** 2026-06-25T13:55:00Z
- **Tasks:** 2
- **Files created:** 1

## Accomplishments

- Added `code/speed_check.py`.
- Implemented real dev/NER sample collection with non-ASCII and double-space coverage.
- Implemented `slow_replay_encode(tokenizer, text)` that replays learned merges without calling public `encode`.
- Added `check_oracle()` to compare public encode against slow replay across 83 samples.
- Added `check_cache()` to compare public encode output against the uncached/oracle path and verify `_encode_cache` is empty after pickle round-trip.
- Added `check_train()` as a baseline hook for Plan 03's incremental trainer checks.

## Task Commits

1. **Task 1: Add speed_check.py oracle harness** - `55b18b3` (`test`)
2. **Task 2: Add cache benchmark placeholder and pickle-state probe** - `55b18b3` (`test`)

**Plan metadata:** pending in close-out commit.

## Verification

- `py -m uv run python code\speed_check.py --only oracle`: PASS.
  - Output: `ORACLE: PASS - 83 samples matched slow replay`
- `py -m uv run python code\speed_check.py --only cache`: PASS.
  - Output included `uncached_seconds=1.187329`, `public_seconds=0.182641`, `CACHE: PASS`.
- `py -m uv run python code\align_check.py`: PASS for VP1, VP2, VP3, VP4, VP6.
- `py -m py_compile code\speed_check.py`: PASS.

## Deviations from Plan

None.

## Issues Encountered

None.

## Next Phase Readiness

Ready for `02-02`: add `_encode_uncached`, wire public `encode` through `_encode_cache`, and tighten `speed_check.py --only cache` to require cached encode faster than uncached encode.

## Self-Check: PASSED

- Slow replay oracle exists outside public encode.
- Cache/pickle probe exists and passes.
- Phase 1 alignment gates remain green.

---
*Phase: 02-fast-encode-training-efficiency*
*Completed: 2026-06-25*
