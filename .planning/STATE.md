---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 4
current_phase_name: Tokenizer 3 Hidden-Domain Generalization
status: ready_to_plan
stopped_at: Phase 4 context gathered; next is planning Phase 4
last_updated: "2026-06-26T00:00:00.000Z"
last_activity: 2026-06-26
last_activity_desc: Closed 03-03 — tok1 vocab 8000 and tok2 vocab 16000 selected and staged
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 10
  completed_plans: 10
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-24)

**Core value:** Three correct, reproducible BPE tokenizers that pass every hard gate and score well on the competition's three axes (efficiency, speed, F1), including the hidden domain.
**Current focus:** Phase 4 - Tokenizer 3 Hidden-Domain Generalization

## Current Position

Phase: 4 of 6 (Tokenizer 3 Hidden-Domain Generalization) - Phase 3 complete
Plan: Phase 3 closed (3/3); Phase 4 not yet planned
Status: Ready to plan Phase 4
Last activity: 2026-06-26 - Closed 03-03 (tok1 vocab 8000, tok2 vocab 16000 selected and staged)

Progress: [#####-----] 50% (3 of 6 phases complete; all planned plans done)

## Performance Metrics

**Velocity:**

- Total plans completed: 10
- Average duration: 23 min
- Total execution time: 2.3 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 4 | 4 | 30 min |
| 2 | 3 | 3 | -- |
| 3 | 3 | 3 | -- |

**Recent Trend:**

- Last 5 plans: 02-02 (10 min), 02-03, 03-01, 03-02, 03-03
- Trend: Phase 3 tok1/tok2 tuning is closed; next risk is hidden-domain tok3 generalization

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
- [03-01]: Candidate tok1/tok2 pickles were generated only through unedited `generate_tokenizers.py` for vocab sizes 4000, 8000, and 16000; `code/tune_tokenizers.py` measures CPU efficiency/speed without training.
- [03-02]: VM F1 sweep completed on Tesla M60. Tok1 F1: 4000=0.4237, 8000=0.4352, 16000=0.3692. Tok2 F1: 4000=0.9610, 8000=0.9570, 16000=0.9557.
- [03-03]: Selected tok1 vocab 8000 (best F1) and tok2 vocab 16000 (within 0.0053 F1 of best while best efficiency/speed); staged measured candidates into `trained_tokenizers/`.

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
Stopped at: Phase 4 context gathered; next is planning Phase 4
Resume file: .planning/phases/04-tokenizer-3-hidden-domain-generalization/04-CONTEXT.md
