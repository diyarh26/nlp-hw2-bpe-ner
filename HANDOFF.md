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

- **Last updated by:** Codex - 2026-06-25 (after commit `6e3656a`; plus this HANDOFF commit)
- **Phase:** 2 of 6 - Fast Encode + Training Efficiency
- **Phase status:** Ready for planning. Phase 1 is complete: 4/4 plans closed.
- **Code state:** `code/bpe_tokenizer.py` is a gates-first, pickle-safe BPE tokenizer with char-up training, mergeable space marker, one-internal-space cap, forced legal bigram, byte fallback, and exact decode reconstruction. `TRAIN_TEXT_LIMIT = 1000` is a temporary Phase 1 feasibility cap.
- **Validation state:** Phase 1 fallback submission exists locally as ignored `HW2_214034621.zip`; it passes `check_submission.py` with no `[ERROR]` when run with `PYTHONIOENCODING=utf-8`. Local CPU NER on domain 1 reached best dev F1 `0.4218`.

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

Plan and execute **Phase 2: Fast Encode + Training Efficiency**.

Start by creating Phase 2 planning artifacts under `.planning/phases/02-fast-encode-training-efficiency/`, then implement the first safe slice. The roadmap success criteria are:

- `encode` uses a merge-rank dict / greedy lowest-rank strategy and stays byte-identical to the Phase 1 encoder on dev samples.
- Per-chunk encode cache measurably reduces repeated-segment encode time.
- Training uses weighted word-frequency chunks plus incremental pair-count deltas with an inverted `pair -> chunks` index, replacing the temporary Phase 1 `TRAIN_TEXT_LIMIT = 1000` reliance.
- All Phase 1 gates still pass after the rewrite: `align_check.py`, fresh pickle load/reproducibility, forbidden hash check, and checker-clean fallback zip.

Read before editing:

- `.planning/ROADMAP.md` Phase 2
- `.planning/REQUIREMENTS.md` SPEED-01/SPEED-02/SPEED-03
- `.planning/phases/01-gates-first-submittable-tokenizer/01-04-SUMMARY.md`
- `code/bpe_tokenizer.py`
- `code/align_check.py`
- `generate_tokenizers.py`
- `check_submission.py`

Useful verification commands:

```powershell
py -m uv run python code/align_check.py
$env:PYTHONIOENCODING='utf-8'; py -m uv run python check_submission.py HW2_214034621.zip
```

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
- **Current tokenizer note:** Phase 1 intentionally uses `TRAIN_TEXT_LIMIT = 1000`; Phase 2 should replace this with efficient training over weighted chunks rather than treating the cap as final.
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
