# HANDOFF - Agent Relay Baton

> Single source of truth for continuing this project across agents (Claude <-> Codex).
> Read this FIRST. Update it LAST before stopping. Commit it. The git history is our shared memory.

## Relay protocol

1. **On start:** if a git remote exists, `git pull`. Then read this file top-to-bottom, especially **NEXT ACTION**, then open the files it points to.
2. **Work the next chunk.** Commit early and often with clear, conventional messages (`feat(...)`, `fix(...)`, `docs(...)`). Small commits let the other agent see exactly what changed.
3. **Before stopping:** update **Current status**, **What was just done**, **NEXT ACTION**, and **Open questions / caveats**. Leave the tree clean, or write a `WIP:` note here with exactly what is half-done. Commit it, and push if a remote exists.
4. **Hand back the baton:** the user pastes the other agent its continuation prompt at the bottom of this file.

## Project

From-scratch **BPE tokenizer** (`code/bpe_tokenizer.py`, `BPETokenizer(BaseTokenizer)`) feeding a **fixed** binary NER model (word-level F1). Deliver **3 trained tokenizers + a <=1-page report** as `HW2_214034621.zip`. Graded NLP homework (HW2). This is **not** a retrieval/search/ranking system.

## Current status

- **Last updated by:** Codex - 2026-06-26 (after Phase 3 planning draft)
- **Phase:** 3 of 6 - Per-Domain Tuning (Tokenizers 1 & 2)
- **Phase status:** **Phase 3 draft planned** under `.planning/phases/03-per-domain-tuning/`; awaiting Claude review before execution.
- **Code state:** `code/bpe_tokenizer.py` = cached encode + **uncapped** full-corpus trainer. `train()` -> `_train_merges()`: integer-symbol incremental merge loop with local pair-count deltas, a lazy max-heap, and a superset `pair->chunks` inverted index. Byte-identical merges to the full-recount oracle. No `TRAIN_TEXT_LIMIT`.
- **Validation state:** `speed_check.py --only all` PASS (oracle + cache + train==full-recount), `align_check.py` PASS (VP1-4,6). Full `generate_tokenizers.py --vocab_size 8000 --seed 42` completes in ~4.5 min; all 3 pkls load; `check_submission.py HW2_214034621.zip` exits 0 with no `[ERROR]`; tokenizers byte-reproducible across double-runs; forbidden-file hashes intact. `trained_tokenizers/` and `HW2_214034621.zip` are ignored local artifacts (rebuild with the commands below).

## What was just done (most recent first)

- **03 planning draft (Codex):** created `.planning/phases/03-per-domain-tuning/03-CONTEXT.md` plus `03-01-PLAN.md`, `03-02-PLAN.md`, and `03-03-PLAN.md`. The draft splits Phase 3 into CPU-local sweep work, VM-only F1 scoring, and selection/regression. It proposes vocab grid `[4000, 8000, 16000]`, `code/tune_tokenizers.py` as the sweep harness, six VM F1 runs with a <=45 min/run assumption, and a knee rule of smallest vocab within 0.02 absolute F1 of best, tie-broken by lower tokens/char then higher tokens/sec. No code or VM/NER execution was performed.
- **02-03 (Phase 2 closed):** the weighted word-chunk trainer was correct but still timed out on full domain-1 generation (1.21 M unique chunks; 73% singletons; heap churn + whole-chunk pair recounts). Claude rewrote the merge bookkeeping as `_train_merges` (int symbols, local-neighbour deltas, disciplined lazy heap, superset inverted index) producing identical merges. Domain-1 vocab-8000: ~124 s; full 3-tokenizer generation: ~273 s. Regenerated all 3 pkls, rebuilt the zip (checker-clean), verified byte-reproducibility and forbidden-file integrity. Removed the old `_build_pair_index`/`_apply_indexed_merge`/`_*_sequence_pairs`/`_best_indexed_pair`/`_push_pair_heap` helpers. `train_commands.txt` annotated with the uncapped trainer method.
- **02-02:** cached public `encode` + `_encode_uncached` + `_ids_from_symbols`; cache speed win proven; pickle-safe (cache dropped on pickle).
- **02-01:** `code/speed_check.py` oracle/benchmark harness (encode replay oracle, cache probe, train-vs-full-recount probe).
- **Phase 1 (complete, 01-01..01-04):** scaffold extracted; `uv` env; `align_check.py` (VP1-4,6); char-up BPE with `▁` marker, capped bigram, byte fallback, length-faithful decode; pickle hardening; first submittable zip; local CPU domain-1 NER best F1 was **0.4218** (with the old capped tokenizer).

## NEXT ACTION

**Claude: review Phase 3 plan, then execute.**

Review these new draft files first:

- `.planning/phases/03-per-domain-tuning/03-CONTEXT.md`
- `.planning/phases/03-per-domain-tuning/03-01-PLAN.md`
- `.planning/phases/03-per-domain-tuning/03-02-PLAN.md`
- `.planning/phases/03-per-domain-tuning/03-03-PLAN.md`

Items deliberately marked for confirmation in `03-CONTEXT.md`:

- Vocab grid `[4000, 8000, 16000]`.
- VM runtime assumption: <=45 minutes per candidate run, <=4.5 hours for six runs plus overhead.
- Selection knee rule: smallest vocab within 0.02 absolute F1 of best, then lower tokens/char, then higher tokens/sec.
- Whether to update `train_commands.txt` during Phase 3 or defer final command freeze to Phase 5.

## Hard rules

- **Never edit** the four forbidden files (kept byte-identical; hashes in `.planning/phases/01-gates-first-submittable-tokenizer/forbidden_files.sha256`):
  - `code/base_tokenizer.py`, `code/train_ner_model.py`, `generate_tokenizers.py`, `check_submission.py`
- Locked stack only: `numpy`, `regex` (`import regex`, not `re`), `torch`, `tqdm`. No new dependencies.
- `decode` stays pure/length-preserving: ids -> token strings -> join -> replace `▁` with real spaces. No strip/lower/normalization.
- Guarantee >=1 cross-word bigram and no token spanning 3+ words. `encode` never emits `[UNK]` id 1.
- Pickle-safe: no lambdas/closures/local classes/`defaultdict(lambda)` in persisted state; keep the explicit `__getstate__`/`__setstate__`.
- If you change `train`/`encode`, keep `speed_check.py --only all` (oracle + train==full-recount) and `align_check.py` green, and keep training deterministic (byte-reproducible).
- GPU only on the Tesla M60 VM (Phase 3 F1 sweep onward). Protect the 10h budget; CPU gates green first.

## Open questions / caveats

- **Git remote:** none configured. Local relay works in this same folder. A remote is only needed if Codex runs on the VM (cross-machine); the ~95MB data zip stays gitignored.
- **Trainer note:** `TRAIN_TEXT_LIMIT` is gone; full generation is feasible (~4.5 min for all 3). If Phase 3 changes the trainer, re-confirm `speed_check.py --only train` (oracle equality) and a small double-run reproducibility check.
- **uv PATH:** run Python via `py -m uv run python ...`.
- **Checker encoding:** on Windows set `$env:PYTHONIOENCODING='utf-8'` before `check_submission.py` (otherwise printing `▁` can fail even when the submission is valid).
- **Ignored deliverables:** `trained_tokenizers/`, `models/`, `HW2_214034621.zip` are gitignored. Rebuild: `py -m uv run python generate_tokenizers.py --vocab_size 8000 --seed 42 --output_dir trained_tokenizers` then `py -m uv run python code/build_submission.py`.
- **GSD helper:** local `gsd-tools.cjs` may fail (missing package metadata); state/roadmap updates can be done by hand.
- **Line endings:** Git reports LF->CRLF warnings on Windows; no functional impact.

## Key files map

- Relay baton: `HANDOFF.md`
- Rules / constraints: `.claude/CLAUDE.md`
- Spec & roadmap: `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, `.planning/STATE.md`
- Research: `.planning/research/{STACK,ARCHITECTURE,PITFALLS,FEATURES,SUMMARY}.md`
- Phase 1 / Phase 2 artifacts: `.planning/phases/01-*/`, `.planning/phases/02-*/`
- Tokenizer: `code/bpe_tokenizer.py`; gates: `code/align_check.py`, `code/speed_check.py`; harness to extend: `code/test_tokenizer.py`

## Continuation prompts

### To Codex
```text
Continue this project from HANDOFF.md (a from-scratch BPE tokenizer for NER, NOT a retrieval/ranking system). Do the NEXT ACTION exactly. Never edit the four forbidden files. Keep speed_check.py --only all and align_check.py green. Do NOT invent a local GPU/NER run -- the VM F1 work is dispatched separately. Commit each completed task and update HANDOFF.md last. If unsure, say so instead of guessing.
```

### To Claude
```text
Read HANDOFF.md and continue from the NEXT ACTION.
```
