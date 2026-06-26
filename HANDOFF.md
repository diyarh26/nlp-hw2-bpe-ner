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

- **Last updated by:** Codex - 2026-06-26 (after Phase 3 VM F1 sweep)
- **Phase:** 3 of 6 - Per-Domain Tuning (Tokenizers 1 & 2)
- **Phase status:** **03-01 complete locally** and **03-02 VM F1 sweep complete**. Final selection/staging (03-03) is intentionally not run yet.
- **Code state:** `code/bpe_tokenizer.py` unchanged from Phase 2: cached encode + **uncapped** full-corpus trainer. `code/tune_tokenizers.py` is now a Phase 3 measure-only harness; it loads generator-produced candidates and writes CPU metrics, but never trains tokenizers.
- **Validation state:** Candidate pkls were produced by unedited `generate_tokenizers.py` for vocab sizes 4000/8000/16000, copied to `tuning/tokenizers/`, and all six load via `BaseTokenizer.load`. `tuning/results/cpu_sweep.csv` has 6 valid rows, positive metrics, and reconstruction rate 1.0. VM F1 logs live in `tuning/logs/vm_f1/`, with summary rows in `tuning/results/vm_f1_results.csv`. Candidate copy hashes match their generator outputs. `speed_check.py --only all` PASS, `align_check.py` PASS, forbidden-file hashes intact from the prior handoff; no code files were changed during the VM sweep.

## What was just done (most recent first)

- **03-02 Task 2 VM F1 sweep (Codex):** cloned the private repo on the Tesla M60 VM, installed missing `uv`, ran `uv sync`, and confirmed CUDA: `True Tesla M60`. Timed the required first run (`tokenizer_1_vocab8000` on `train_1/dev_1_binary.tagged`): Best F1 **0.4352**, wall-time **0.9443 min**, so the stop-rule passed (`0.9443 x 6 ~= 5.7 min`, below the 4.5 h cap). Completed the remaining five runbook commands and wrote `tuning/results/vm_f1_results.csv`: tok1 vocab4000 **0.4237 / 0.8820 min**, tok1 vocab8000 **0.4352 / 0.9443 min**, tok1 vocab16000 **0.3692 / 0.8217 min**; tok2 vocab4000 **0.9610 / 3.6005 min**, tok2 vocab8000 **0.9570 / 3.3208 min**, tok2 vocab16000 **0.9557 / 3.2012 min**. Captured logs under `tuning/logs/vm_f1/`. Did not edit forbidden files, change NER hyperparameters, or run 03-03 selection/staging.
- **Phase 3 review + VM transfer prep (Claude):** reviewed the generator-backed Phase 3 revision and **approved** it (SERDE-03 closed: `tune_tokenizers.py` is measure-only; 03-03 enforces a `fc /b` byte-identity cmp of staged tok1/tok2 vs a fresh `generate_tokenizers.py` run before shipping). CPU sweep verified clean (6 rows, reconstruction 1.0, monotonic efficiency curve). Found the dispatch blocker: no git remote + candidate pkls/tagged data gitignored, so the VM had no way to get them. Per user, going via GitHub: force-added the 6 candidate pkls + 4 `data/ner_data/*.tagged` files (`ea159d4`) and added `TASKS.md` as the repo/relay landing page. **Pushed to private repo `diyarh26/nlp-hw2-bpe-ner` (origin/master).** VM-Codex dispatch prompt handed to the user.
- **03-01 + 03-02 Task 1 (Codex):** revised Phase 3 plans for Claude's approved reproducibility fix. Candidate pkls now come only from unedited `generate_tokenizers.py`; `code/tune_tokenizers.py` is measure-only. Generated candidates for vocab grid `[4000, 8000, 16000]` into `tuning/_gen/{V}` and copied tok1/tok2 to `tuning/tokenizers/` (generation wall-times: 4000 = 200.1s, 8000 = 212.8s, 16000 = 241.6s). Ran the CPU sweep and wrote `tuning/results/cpu_sweep.csv`. Added `.planning/phases/03-per-domain-tuning/03-VM-F1-RUNBOOK.md` with six exact VM commands, first-run tok1@8000 timing step, logs, budget assumption, and stop-rule. Did not run VM/NER and did not run 03-03 staging.
- **03 planning draft (Codex):** created `.planning/phases/03-per-domain-tuning/03-CONTEXT.md` plus `03-01-PLAN.md`, `03-02-PLAN.md`, and `03-03-PLAN.md`. The draft split Phase 3 into CPU-local sweep work, VM-only F1 scoring, and selection/regression.
- **02-03 (Phase 2 closed):** the weighted word-chunk trainer was correct but still timed out on full domain-1 generation (1.21 M unique chunks; 73% singletons; heap churn + whole-chunk pair recounts). Claude rewrote the merge bookkeeping as `_train_merges` (int symbols, local-neighbour deltas, disciplined lazy heap, superset inverted index) producing identical merges. Domain-1 vocab-8000: ~124 s; full 3-tokenizer generation: ~273 s. Regenerated all 3 pkls, rebuilt the zip (checker-clean), verified byte-reproducibility and forbidden-file integrity. Removed the old `_build_pair_index`/`_apply_indexed_merge`/`_*_sequence_pairs`/`_best_indexed_pair`/`_push_pair_heap` helpers. `train_commands.txt` annotated with the uncapped trainer method.
- **02-02:** cached public `encode` + `_encode_uncached` + `_ids_from_symbols`; cache speed win proven; pickle-safe (cache dropped on pickle).
- **02-01:** `code/speed_check.py` oracle/benchmark harness (encode replay oracle, cache probe, train-vs-full-recount probe).
- **Phase 1 (complete, 01-01..01-04):** scaffold extracted; `uv` env; `align_check.py` (VP1-4,6); char-up BPE with `▁` marker, capped bigram, byte fallback, length-faithful decode; pickle hardening; first submittable zip; local CPU domain-1 NER best F1 was **0.4218** (with the old capped tokenizer).

## NEXT ACTION

**Claude reviews Phase 3 selection (03-03): apply the knee rule to cpu_sweep.csv + vm_f1_results.csv.**

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

- **Git remote:** `origin` = private GitHub `diyarh26/nlp-hw2-bpe-ner` (branch `master`). VM-Codex pulls from here. The 6 candidate pkls + 4 tagged files were force-added (gitignored otherwise; untrack with `git rm --cached` after the sweep). The ~95MB raw corpus stays gitignored — not needed for the F1 sweep.
- **Phase 3 status:** CPU sweep and VM F1 sweep are complete; selection/staging is not. Do not run `03-03` until Claude reviews the selection.
- **Trainer note:** `TRAIN_TEXT_LIMIT` is gone; full generation is feasible (~3.3-4.0 min per full 3-tokenizer candidate run in Phase 3). If Phase 3 changes the trainer, re-confirm `speed_check.py --only train` (oracle equality) and a small double-run reproducibility check.
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
