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

- **Last updated by:** Codex - 2026-06-26 (after commit `5d4945e`; plus this HANDOFF commit)
- **Phase:** 2 of 6 - Fast Encode + Training Efficiency
- **Phase status:** In progress - `02-01` and `02-02` complete; `02-03` Task 1 committed; `02-03` Task 2 is WIP/blocked on full generation speed.
- **Code state:** `code/bpe_tokenizer.py` now has cached encode plus a weighted incremental trainer. `TRAIN_TEXT_LIMIT` has been removed. The current trainer passes local small-fixture oracle checks but is still too slow for full domain-1 vocab-8000 generation.
- **Validation state:** `py -m uv run python code\speed_check.py --only all` and `py -m uv run python code\align_check.py` passed after commit `5d4945e`. Full `generate_tokenizers.py --vocab_size 8000 --seed 42 --output_dir trained_tokenizers` has NOT completed; `trained_tokenizers/` is currently empty.

## WIP: 02-03 Task 2 Blocked

Attempted the required full generation on 2026-06-26:

```powershell
py -m uv run python generate_tokenizers.py --vocab_size 8000 --seed 42 --output_dir trained_tokenizers
```

Result: timed out after 30 minutes. No `trained_tokenizers/tokenizer_1.pkl` was written, so it was still in tokenizer 1 / domain 1 training. The process tree survived the timeout and was manually stopped. `trained_tokenizers/` is empty.

Quick diagnostics:

- `domain_1_train.txt`: 1,280,508 lines, 1,266,678 unique lines, 95,082,468 chars, average length 74.25.
- `domain_2_train.txt`: 319,707 lines, 316,459 unique lines, 29,268,720 chars, average length 91.55.
- A bounded full-domain `domain_1` run at `vocab_size=1000` also timed out after about 3 minutes before producing output.
- A 20k-line sampled cProfile at `vocab_size=1000` also timed out after about 3 minutes.
- A 2k-line sampled cProfile at `vocab_size=700` completed in 51.776 seconds for 324 merges. Top cumulative hotspots:
  - `_apply_indexed_merge`: 35.070s
  - `_remove_sequence_pairs`: 17.272s
  - `_best_indexed_pair`: 16.171s
  - `_add_sequence_pairs`: 15.341s
  - `_sequence_pair_counts`: 14.390s
  - `_heapq.heappop`: 13.304s
  - `_push_pair_heap`: 10.282s

Interpretation: weighted unique texts barely reduce domain 1 because almost every line is unique. The current indexed trainer avoids full global recounts, but it still performs too much heap churn and repeated full sequence pair recounting for impacted sequences. Do not reintroduce a training-data cap as the normal path; optimize the incremental data structures instead.

## What was just done

Executed and closed Phase 1 plans `01-03` and `01-04` after the earlier `01-01` / `01-02` work.

`01-01`:

- Extracted `code_env_example.zip` selectively, excluding `HW2_123456789.zip` and `code/__pycache__/`.
- Extracted `data (3).zip` to ignored `data/`.
- Added `.gitignore` for local env/data/training artifacts and assignment source files.
- Recorded `.planning/phases/01-gates-first-submittable-tokenizer/forbidden_files.sha256` for:
  - `code/base_tokenizer.py`
  - `code/train_ner_model.py`
  - `generate_tokenizers.py`
  - `check_submission.py`
- Installed `uv` with `py -m pip install --user uv`, then ran `py -m uv sync`.
- Verified imports under `uv run`: `numpy 2.4.6`, `regex 2026.5.9`, `torch 2.12.0+cu126`, `tqdm 4.68.1`.
- Added `code/align_check.py`, covering VP-1, VP-2, VP-3, VP-4, and VP-6.
- Created `.planning/phases/01-gates-first-submittable-tokenizer/01-01-SUMMARY.md`.
- Updated `.planning/STATE.md` and `.planning/ROADMAP.md`.

`01-02`:

- Implemented `BPETokenizer.train()` in `code/bpe_tokenizer.py`:
  - deterministic char-up BPE
  - sorted base vocabulary
  - mergeable `▁` space marker
  - one-internal-space merge cap
  - forced legal cross-word bigram fallback
  - `TRAIN_TEXT_LIMIT = 5000`
- Implemented `BPETokenizer.encode()` with private-use byte-token fallback so `[UNK]` id 1 is never emitted.
- Implemented `BPETokenizer.decode()` with exact reconstruction for normal chars, spaces, and fallback UTF-8 bytes.
- Verified:
  - `py -m uv run python code/align_check.py --only vp1,vp2,vp4,vp6` passes.
  - `py -m uv run python code/align_check.py` also passes, including VP-3.
  - All 256 byte fallback tokens are present after training.
  - Forbidden-file hashes still match baseline.
- Created `.planning/phases/01-gates-first-submittable-tokenizer/01-02-SUMMARY.md`.
- Updated `.planning/STATE.md`, `.planning/ROADMAP.md`, and `.planning/REQUIREMENTS.md`.

`01-03`:

- Added explicit `BPETokenizer.__getstate__()` / `__setstate__()` and runtime helper rebuilding.
- Stored the regex pattern as a primitive string and excluded runtime cache/compiled objects from pickle state.
- Fixed forced-bigram fallback to preserve the requested `vocab_size` cap.
- Proved all three generated tokenizers are byte-reproducible across repeated unedited `generate_tokenizers.py --vocab_size 600` runs.
- Created `.planning/phases/01-gates-first-submittable-tokenizer/01-03-SUMMARY.md`.

`01-04`:

- Reduced `TRAIN_TEXT_LIMIT` to 1000 so Phase-1 `--vocab_size 8000` generation completes locally.
- Generated all three `trained_tokenizers/tokenizer_{1,2,3}.pkl` through the unedited `generate_tokenizers.py` default mapping.
- Recorded the generator command and default mapping in `train_commands.txt`.
- Ran the full unedited local CPU NER pipeline on domain 1 with tokenizer 1; best dev F1 was `0.4218`.
- Added `code/build_submission.py`, which writes a minimal valid placeholder `report_214034621.pdf` and builds the exact required zip layout.
- Built `HW2_214034621.zip`; checker exited 0 and produced no `[ERROR]` with `PYTHONIOENCODING=utf-8`.
- Rechecked forbidden-file hashes against `.planning/phases/01-gates-first-submittable-tokenizer/forbidden_files.sha256`.
- Created `.planning/phases/01-gates-first-submittable-tokenizer/01-04-SUMMARY.md`.
- Updated `.planning/ROADMAP.md`, `.planning/STATE.md`, and `.planning/REQUIREMENTS.md` so Phase 1 is complete and Phase 2 is next.

Recent commits added by Codex:

- `e57313e` - `fix(01-03): harden tokenizer serialization`
- `3bd3723` - `docs(01-03): complete serialization reproducibility plan`
- `d441028` - `fix(01-04): generate phase-one tokenizers feasibly`
- `dcc6572` - `chore(01-04): ignore local NER model outputs`
- `4549625` - `feat(01-04): build fallback submission zip`
- `6e3656a` - `docs(01-04): complete fallback submission plan`

## NEXT ACTION

Continue **Phase 2 / 02-03 Task 2** by optimizing the current weighted incremental trainer so full generation completes without reintroducing a normal training-data cap.

Current likely optimization targets:

- Reduce lazy heap churn in `_best_indexed_pair` / `_push_pair_heap`.
- Avoid full `_sequence_pair_counts` recounts for every impacted sequence on every merge; update local neighbor pairs around each replacement instead.
- Consider representing each sequence with mutable token ids/linked neighbors or per-sequence pair positions so impacted updates are local.
- Keep deterministic tie-breaking and Phase 1 gates intact.

After optimizing, rerun:

```powershell
py -m uv run python code\speed_check.py --only all
py -m uv run python code\align_check.py
py -m uv run python generate_tokenizers.py --vocab_size 8000 --seed 42 --output_dir trained_tokenizers
py -m uv run python code\build_submission.py
$env:PYTHONIOENCODING='utf-8'; py -m uv run python check_submission.py HW2_214034621.zip
```

Then close `.planning/phases/02-fast-encode-training-efficiency/02-03-PLAN.md` only if all checks pass.

Read before editing:

- `.planning/phases/02-fast-encode-training-efficiency/02-03-PLAN.md`
- `.planning/phases/02-fast-encode-training-efficiency/02-RESEARCH.md`
- `.planning/phases/02-fast-encode-training-efficiency/02-VALIDATION.md`
- `code/bpe_tokenizer.py`
- `code/speed_check.py`
- `generate_tokenizers.py`
- `code/build_submission.py`
- `check_submission.py`

## Hard rules

- **Never edit** the four forbidden files after extraction:
  - `code/base_tokenizer.py`
  - `code/train_ner_model.py`
  - `generate_tokenizers.py`
  - `check_submission.py`
- Use only the locked stack: `numpy`, `regex`, `torch`, `tqdm`. No new project dependencies.
- Use `import regex`, not stdlib `re`, in tokenizer code if regex is needed.
- `decode` must be pure and length-preserving: ids -> token strings -> join -> replace space marker with real spaces. No strip/lower/normalization.
- Guarantee at least one cross-word bigram and no token spanning 3+ words.
- `encode` must never emit `[UNK]` id 1.
- Keep tokenizer pickle-safe: no lambdas/closures/local classes/defaultdict(lambda) in persisted state; preserve the explicit `__getstate__`/`__setstate__` contract.
- No GPU run until the Phase 5 VM evaluation plan. Protect the 10h Tesla M60 budget.

## Open questions / caveats

- **Git remote:** none configured. Local handoff works in this folder; remote only matters for VM relay.
- **GSD helper issue:** `node $HOME/.codex/gsd-core/bin/gsd-tools.cjs ...` currently fails because `../../../package.json` is missing from the local GSD install. Codex manually updated GSD metadata.
- **uv PATH:** `uv` is installed in the user Python scripts directory. In PowerShell, either prepend `$env:APPDATA\Python\Python311\Scripts` to `PATH`, or run `py -m uv ...`.
- **Checker encoding:** On Windows, set `$env:PYTHONIOENCODING='utf-8'` before `check_submission.py`; otherwise printing the space marker may fail even though the submission is valid.
- **Ignored deliverables:** `trained_tokenizers/`, `models/`, and `HW2_214034621.zip` are ignored local artifacts. Rebuild the zip with `py -m uv run python code/build_submission.py` if these artifacts are missing.
- **Current trainer note:** `TRAIN_TEXT_LIMIT` is gone. Full generation is blocked by trainer performance, not by correctness gates. Do not add a normal cap to make the run pass.
- **Line endings:** Git reports LF-to-CRLF warnings on Windows. No functional issue observed.

## Key files map

- Relay baton: `HANDOFF.md`
- Rules / constraints: `.claude/CLAUDE.md`
- Spec & roadmap: `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, `.planning/STATE.md`
- Research: `.planning/research/{STACK,ARCHITECTURE,PITFALLS,FEATURES,SUMMARY}.md`
- Phase 1 artifacts: `.planning/phases/01-gates-first-submittable-tokenizer/`
- Red harness: `code/align_check.py`
- Tokenizer to implement next: `code/bpe_tokenizer.py`

## Continuation prompts

### To Codex

```text
Continue this project from HANDOFF.md. Do the NEXT ACTION exactly. Never edit the four forbidden files. Commit each completed task and update HANDOFF.md last.
```

### To Claude

```text
Read HANDOFF.md and continue from the NEXT ACTION.
```
