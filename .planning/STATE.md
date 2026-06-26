---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 3
current_phase_name: Per-Domain Tuning (Tokenizers 1 & 2)
status: ready_to_plan
stopped_at: Phase 2 complete (02-03 closed); next is planning Phase 3
last_updated: "2026-06-26T00:00:00.000Z"
last_activity: 2026-06-26
last_activity_desc: Closed 02-03 — uncapped incremental BPE trainer; full-corpus pkls regenerated, checker-clean
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 7
  completed_plans: 7
  percent: 33
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-24)

**Core value:** Three correct, reproducible BPE tokenizers that pass every hard gate and score well on the competition's three axes (efficiency, speed, F1), including the hidden domain.
**Current focus:** Phase 2 - Fast Encode + Training Efficiency

## Current Position

Phase: 3 of 6 (Per-Domain Tuning) — Phase 2 complete
Plan: Phase 2 closed (3/3); Phase 3 not yet planned
Status: Ready to plan Phase 3
Last activity: 2026-06-26 - Closed 02-03 (uncapped incremental trainer; full-corpus pkls, checker-clean)

Progress: [###-------] 33% (2 of 6 phases complete; all planned plans done)

## Performance Metrics

**Velocity:**

- Total plans completed: 6
- Average duration: 23 min
- Total execution time: 2.3 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 4 | 4 | 30 min |
| 2 | 3 | 3 | -- |

**Recent Trend:**

- Last 5 plans: 01-02 (7 min), 01-03 (14 min), 01-04 (82 min), 02-01 (8 min), 02-02 (10 min)
- Trend: Phase 1 hard gates are green; Phase 2 encode speed work is complete

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Gates-first vertical MVP - Phase 1 ships a fully submittable artifact (all hard gates + F1>=0.4 on one domain) before any competition tuning.
- [Roadmap]: Cross-word BPE merges with a SentencePiece-style mergeable space marker, capped at one internal space (<= bigram).
- [Roadmap]: Build/test tokenizer locally on CPU; gate the GPU (Tesla M60, 10h budget) NER evaluation behind all-local-gates-green.
- [01-01]: Use `code/align_check.py` as the red local gate; it loads real data fixtures and fails cleanly until `BPETokenizer.train/encode/decode` are implemented.
- [01-01]: `bash init.sh` timed out locally; the working local env was created with Windows-native `py -m uv sync`.
- [01-02]: `BPETokenizer` now implements deterministic char-up BPE, the `▁` space marker, one-internal-space merge cap, forced bigram fallback, zero-UNK byte fallback, and exact decode reconstruction.
- [01-03]: `BPETokenizer` now has explicit primitive-only pickle state; all three generated tokenizer pickles were byte-identical across two unedited generator runs at `--vocab_size 600`.
- [01-04]: Phase-1 fallback tokenizers are generated at `--vocab_size 8000`; local CPU domain-1 NER best F1 was 0.4218; `HW2_214034621.zip` passes `check_submission.py` with no `[ERROR]`.
- [02-plan]: Phase 2 is split into three plans: oracle/benchmark harness, cached encode rewrite, and weighted incremental trainer plus checker regression.
- [02-01]: `code/speed_check.py` now provides slow replay encode oracle, cache/pickle probe, and baseline training probe.
- [02-02]: `BPETokenizer.encode` now uses a runtime-only full-text cache with `_encode_uncached` and `_ids_from_symbols`; cache benchmark shows a clear repeated-call speed win.
- [02-03]: Full-corpus training enabled via an integer-symbol incremental merge loop (`_train_merges`: local pair-count deltas + lazy heap + superset `pair->chunks` index), byte-identical to the full-recount oracle. Domain-1 vocab-8000 trains in ~2 min; all 3 pkls regenerated, checker-clean, byte-reproducible. `TRAIN_TEXT_LIMIT` removed. (Trainer rewrite done by Claude after Codex's chunk-dedup WIP still timed out.)

### Pending Todos

[From .planning/todos/pending/ - ideas captured during sessions]

None yet.

### Blockers/Concerns

[Issues that affect future work]

- Decode length-faithfulness is a silent F1 linchpin - keep `code/align_check.py` green after Phase 2 rewrites.
- 10h VM budget is hard - no GPU run until Phase 1 local gates are green; subsequent VM use only in Phase 5.
- Local `gsd-tools.cjs` currently fails due to missing package metadata (`../../../package.json`); state and roadmap updates were done manually for 01-01.
- [RESOLVED 02-03] `TRAIN_TEXT_LIMIT` removed; the weighted word-chunk incremental trainer trains the full corpora (domain-1 vocab-8000 ~2 min, full 3-tokenizer generation ~4.5 min). No training-data cap remains.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Optimization | OPT-01/02/03 (heap encode, merge regularization, Twitter preprocessing) | v2 - pursue only if v1 solid and budget allows | 2026-06-24 |

## Session Continuity

Last session: 2026-06-26
Stopped at: Phase 2 complete (02-03 closed); next is planning Phase 3
Resume file: HANDOFF.md (NEXT ACTION = plan Phase 3)
