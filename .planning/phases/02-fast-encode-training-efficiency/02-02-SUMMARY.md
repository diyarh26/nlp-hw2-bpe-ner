---
phase: 02-fast-encode-training-efficiency
plan: 02
subsystem: tokenizer
tags: [encode, cache, speed, oracle]
requires:
  - phase: 02-01
    provides: Encode oracle and cache benchmark harness
provides:
  - Cached public encode path
  - Uncached encode helper for benchmarking
  - Symbol-to-id helper with byte fallback
affects:
  - 02-03
tech-stack:
  added: []
  patterns:
    - Runtime-only full-text encode cache
    - Defensive list copies from cache
    - Uncached helper used by benchmark harness
key-files:
  created: []
  modified:
    - code/bpe_tokenizer.py
    - code/speed_check.py
key-decisions:
  - "Cache exact input strings as full encode results to preserve cross-word merge behavior."
  - "Return defensive copies from the cache so callers cannot mutate runtime cache state."
  - "Use `_encode_uncached` as the stable benchmark path for Phase 2 verification."
patterns-established:
  - "`BPETokenizer.encode` checks `_encode_cache` before computing ids."
  - "`BPETokenizer._encode_uncached` bypasses cache for oracle/benchmark checks."
requirements-completed:
  - SPEED-01
  - SPEED-02
duration: 10 min
completed: 2026-06-25
status: complete
---

# Phase 2 Plan 2: Cached Encode Summary

**The tokenizer now has a cached public encode path with an explicit uncached helper, and the Phase 2 oracle proves behavior stayed byte-identical.**

## Performance

- **Duration:** 10 min
- **Completed:** 2026-06-25T14:06:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Refactored `BPETokenizer.encode(text)` to check `_encode_cache` by exact input string.
- Added `BPETokenizer._encode_uncached(text)` for benchmarking and non-cache oracle comparisons.
- Added `BPETokenizer._ids_from_symbols(symbols)` to centralize id lookup and byte fallback.
- Updated `code/speed_check.py --only cache` to require cached encode faster than uncached encode.
- Preserved pickle safety: `_encode_cache` is still rebuilt empty after load and is not serialized.

## Task Commits

1. **Task 1: Add uncached encode helper and public cache** - `b8946d9` (`feat`)
2. **Task 2: Tighten cache benchmark to require a speed win** - `b8946d9` (`feat`)

**Plan metadata:** pending in close-out commit.

## Verification

- `py -m uv run python code\speed_check.py --only oracle`: PASS.
  - Output: `ORACLE: PASS - 83 samples matched slow replay`
- `py -m uv run python code\speed_check.py --only cache`: PASS.
  - Output: `uncached_seconds=1.945941`, `cached_seconds=0.008744`, `cache_entries=13`, `CACHE: PASS`.
- `py -m uv run python code\speed_check.py --only all`: PASS.
  - Output included `uncached_seconds=1.660482`, `cached_seconds=0.009379`, and `TRAIN: PASS`.
- `py -m uv run python code\align_check.py`: PASS for VP1, VP2, VP3, VP4, VP6.
- `py -m py_compile code\bpe_tokenizer.py code\speed_check.py`: PASS.

## Deviations from Plan

None.

## Issues Encountered

None.

## Next Phase Readiness

Ready for `02-03`: replace the capped full-recount training path with weighted unique sequences and incremental pair-index updates, then regenerate tokenizers and rerun the checker gate.

## Self-Check: PASSED

- Public encode is cached and returns defensive copies.
- `_encode_uncached` exists and bypasses cache.
- Cache is runtime-only and pickle-dropped.
- Oracle equality and alignment gates remain green.

---
*Phase: 02-fast-encode-training-efficiency*
*Completed: 2026-06-25*
