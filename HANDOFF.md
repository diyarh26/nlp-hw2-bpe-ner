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

- **Last updated by:** Codex - 2026-06-25 (after commit `c969296`; plus this HANDOFF commit)
- **Phase:** 1 of 6 - Gates-First Submittable Tokenizer
- **Phase status:** In progress - `01-01` and `01-02` complete, `01-03` next. Progress: 2/4 plans.
- **Code state:** `code/` scaffold extracted. `data/` extracted and gitignored. `.venv/` created by Windows-native `uv sync` and gitignored.
- **Validation state:** `code/align_check.py` is green for VP-1, VP-2, VP-3, VP-4, and VP-6. VP-3 is ahead of schedule, but `01-03` still owns explicit serialization/reproducibility hardening.

## What was just done

Executed Phase 1 plans `01-01` and `01-02`.

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

Commits added by Codex:

- `9cecf5f` - `chore(01-01): extract scaffold and data baseline`
- `5ac8461` - `test(01-01): add red tokenizer alignment harness`
- `f919fd7` - `docs(01-01): complete scaffold harness plan`
- `7c28e47` - `feat(01-02): implement gates-first BPE tokenizer`
- `c969296` - `docs(01-02): complete tokenizer core plan`

## NEXT ACTION

Execute **`01-03`**:

`.planning/phases/01-gates-first-submittable-tokenizer/01-03-PLAN.md`

Goal: add explicit serialization hardening (`__getstate__`/`__setstate__`, cache/regex handling if needed) and prove byte-reproducibility across two unedited `generate_tokenizers.py` runs.

Use this command pattern for verification in PowerShell:

```powershell
$env:Path = "$env:APPDATA\Python\Python311\Scripts;$env:Path"
uv run python code/align_check.py
```

Or use the module form, which does not depend on PATH:

```powershell
py -m uv run python code/align_check.py
```

Read before editing:

- `.planning/phases/01-gates-first-submittable-tokenizer/01-03-PLAN.md`
- `.planning/phases/01-gates-first-submittable-tokenizer/01-CONTEXT.md`
- `.planning/phases/01-gates-first-submittable-tokenizer/01-RESEARCH.md`
- `.planning/phases/01-gates-first-submittable-tokenizer/01-VALIDATION.md`
- `code/base_tokenizer.py`
- `code/bpe_tokenizer.py`
- `code/train_ner_model.py`
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
- Keep tokenizer pickle-safe: no lambdas/closures/local classes/defaultdict(lambda) in persisted state; `01-03` should make this explicit with `__getstate__`/`__setstate__`.
- No GPU run until local gates are green. Protect the 10h Tesla M60 budget.

## Open questions / caveats

- **Git remote:** none configured. Local handoff works in this folder; remote only matters for VM relay.
- **GSD helper issue:** `node $HOME/.codex/gsd-core/bin/gsd-tools.cjs ...` currently fails because `../../../package.json` is missing from the local GSD install. Codex manually updated `STATE.md` and `ROADMAP.md` for `01-01`.
- **uv PATH:** `uv` is installed in the user Python scripts directory. In PowerShell, either prepend `$env:APPDATA\Python\Python311\Scripts` to `PATH`, or run `py -m uv ...`.
- **Current tokenizer note:** `BPETokenizer` currently has no compiled regex or cache, so VP-3 already passes. `01-03` should still implement the planned explicit state protocol and byte-reproducibility proof.
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
