---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 6
current_phase_name: Report & Final Packaging
status: complete
stopped_at: Submission finalized; no remaining planned work
last_updated: "2026-06-27T18:35:00.000+03:00"
last_activity: 2026-06-27
last_activity_desc: Final report written, submission zip rebuilt, checker/align/speed/forbidden hashes green
progress:
  total_phases: 6
  completed_phases: 6
  total_plans: 16
  completed_plans: 16
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-24)

**Core value:** Three correct, reproducible BPE tokenizers that pass every hard gate and score well on the competition's three axes (efficiency, speed, F1), including the hidden domain.
**Current focus:** Submission finalized - report and checker-valid zip are ready.

## Current Position

Phase: 6 of 6 (Report & Final Packaging) - complete
Plan: Final report and package close-out complete
Status: Final `HW2_214034621.zip` is checker-valid; align/speed/forbidden hashes green
Last activity: 2026-06-27 - Wrote `report_214034621.pdf`, rebuilt zip, and verified all final gates

Progress: [##########] 100% (6 of 6 phases complete; all planned plans done)

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
- [04-01/02]: Tok3 trains on a deterministic balanced mix (all domain-2 + same-count SHA-256 sample of domain-1, exact-line dedup after mixing, 634,359 lines); 3 generator-produced candidates (vocab 4000/8000/16000); CPU/OOV gates all green (0 UNK, recon 1.0, >=1 bigram, 0 trigram, OOV 15/15).
- [04-03]: Tok3 visible-domain VM F1 decreases with vocab — 4000 d1=0.4450/d2=0.9616, 8000 d1=0.4217/d2=0.9587, 16000 d1=0.3792/d2=0.9585 (larger mix-vocab over-merges).
- [04-04]: Hidden-generalization score top pick was 4000; user selected vocab **8000** (balanced: ~14% better efficiency than 4000 for ~0.013 mean-F1 cost, medium-vocab range). Staged mix-trained tok3@8000; byte-reproducible vs fresh generator; tok1/tok2 preserved; align/speed/build/check all green. GEN-01/02/03 complete.
- [05]: Final tokenizers measured in `tuning/results/final_eval.csv`: tok1 mix@6000 F1=0.4780, tok2 vocab16000 F1=0.9557, tok3 mix@8000 F1=0.4217/0.9587 with efficiency/speed logged.
- [06]: `report_214034621.pdf` was generated as a one-page Arial report with top/bottom merge bigrams, dev F1/efficiency, hidden-domain rationale, and model-2 discussion. `HW2_214034621.zip` rebuilt and `check_submission.py`, `align_check.py`, `speed_check.py --only all`, and forbidden hashes all passed.

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

Last session: 2026-06-27
Stopped at: Final submission complete; no remaining planned work
Resume file: HANDOFF.md
