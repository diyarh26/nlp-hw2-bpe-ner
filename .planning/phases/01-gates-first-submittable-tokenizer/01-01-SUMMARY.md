---
phase: 01-gates-first-submittable-tokenizer
plan: 01
subsystem: infra
tags: [uv, scaffold, data, validation, tokenizer-harness]
requires: []
provides:
  - Extracted course scaffold in repo layout
  - Extracted provided data under data/
  - Forbidden-file SHA-256 baseline
  - Windows-native uv environment
  - Red align_check.py local gate harness
affects:
  - 01-02
  - 01-03
  - 01-04
tech-stack:
  added: []
  patterns:
    - Plain-assert validation harness with no extra test framework
    - Forbidden-file hash baseline before implementation
key-files:
  created:
    - .gitignore
    - .planning/phases/01-gates-first-submittable-tokenizer/forbidden_files.sha256
    - code/align_check.py
    - code/base_tokenizer.py
    - code/bpe_tokenizer.py
    - code/test_tokenizer.py
    - code/train_tokenizer.py
    - code/train_ner_model.py
    - check_submission.py
    - generate_tokenizers.py
    - init.sh
    - pyproject.toml
    - uv.lock
  modified: []
key-decisions:
  - "Used selective zip extraction to exclude the sample submission zip and __pycache__ files."
  - "Used Windows-native py -m uv sync after bash init.sh timed out and left a POSIX-style partial .venv."
  - "Kept PKG-01 pending in REQUIREMENTS.md because the final checker-passing zip is delivered in 01-04."
patterns-established:
  - "align_check.py prints per-gate PASS/FAIL lines and exits non-zero on red gates."
  - "Forbidden files are tracked against .planning/phases/01-gates-first-submittable-tokenizer/forbidden_files.sha256."
requirements-completed:
  - PKG-01
duration: 19 min
completed: 2026-06-25
status: complete
---

# Phase 1 Plan 1: Wave 0 Scaffold and Harness Summary

**Course scaffold, data, uv environment, forbidden-file baseline, and red tokenizer gate harness are ready for BPE implementation.**

## Performance

- **Duration:** 19 min
- **Started:** 2026-06-25T11:29:00Z
- **Completed:** 2026-06-25T11:48:00Z
- **Tasks:** 3
- **Files modified:** 14 tracked files plus ignored local data/env artifacts

## Accomplishments

- Extracted the course scaffold into the expected root layout while excluding `HW2_123456789.zip` and `code/__pycache__/`.
- Extracted the provided corpora into ignored `data/` so local validation and later tokenizer generation can run.
- Created `.planning/phases/01-gates-first-submittable-tokenizer/forbidden_files.sha256` for the four forbidden files.
- Bootstrapped a Windows-native `.venv` with the locked stack: `numpy 2.4.6`, `regex 2026.5.9`, `torch 2.12.0+cu126`, `tqdm 4.68.1`.
- Added `code/align_check.py`, the expected-red local gate harness for VP-1, VP-2, VP-3, VP-4, and VP-6.

## Task Commits

1. **Task 1: Extract scaffold + data and record baseline** - `9cecf5f` (`chore`)
2. **Task 2: Bootstrap uv environment** - no tracked commit; `.venv/` is ignored by design
3. **Task 3: Create failing local gate harness** - `5ac8461` (`test`)

**Plan metadata:** pending in close-out commit.

## Files Created/Modified

- `.gitignore` - Ignores local env, data, trained tokenizers, generated zips, and source assignment artifacts.
- `.planning/phases/01-gates-first-submittable-tokenizer/forbidden_files.sha256` - SHA-256 baseline for `code/base_tokenizer.py`, `code/train_ner_model.py`, `generate_tokenizers.py`, and `check_submission.py`.
- `code/align_check.py` - Plain-assert harness that loads real data fixtures and currently fails cleanly at the unimplemented tokenizer.
- `code/*.py`, `check_submission.py`, `generate_tokenizers.py`, `init.sh`, `pyproject.toml`, `uv.lock`, `README.md` - Course scaffold extracted byte-identically from `code_env_example.zip`.

## Verification

- `py -c "... required file check ..."`: PASS - all scaffold and required data files are present.
- `uv run python -c "import numpy, regex, torch, tqdm; ..."`: PASS - locked dependencies import under `.venv`.
- `uv run python code/align_check.py`: EXPECTED FAIL - fixture loading passes; `BPETokenizer.train()` fails with `NotImplementedError`, which is the red gate for 01-02.
- Forbidden-file hash recheck: PASS - all four forbidden files match the baseline manifest.
- `code/align_check.py` acceptance checks: PASS - >=80 lines, imports `BPETokenizer` from `code/`, contains checker-mirroring `.replace(...).strip().count(" ")` predicate, checks `space_token in tokenizer.token_to_id`, and has no `import re`.

## Decisions Made

- Used selective extraction instead of `Expand-Archive` so unwanted scaffold artifacts never landed in the repo.
- Switched from `bash init.sh` to Windows-native `py -m uv sync` after the bash setup timed out and left a POSIX-style `.venv`; the partial environment was removed only after verifying it was inside the workspace.
- Did not mark `PKG-01` complete in `REQUIREMENTS.md`; this plan prepares PKG-01 infrastructure, while 01-04 actually builds and checks the final zip.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Windows bash setup timed out during uv environment creation**
- **Found during:** Task 2 (Bootstrap uv environment)
- **Issue:** `bash init.sh` timed out after 10 minutes, left a running `bash` process, and created a POSIX-style `.venv` that Windows `uv run` would not use correctly.
- **Fix:** Stopped the stale `bash` process, verified `.venv` was inside the workspace, removed only that partial env, installed `uv` via `py -m pip install --user uv`, then ran `py -m uv sync`.
- **Files modified:** `.venv/` only (ignored)
- **Verification:** `uv run python -c "import numpy, regex, torch, tqdm"` succeeded.
- **Committed in:** no tracked commit; environment artifact is intentionally ignored.

**2. [Rule 3 - Blocking] GSD helper shim unavailable**
- **Found during:** Preflight
- **Issue:** `node $HOME/.codex/gsd-core/bin/gsd-tools.cjs` failed because `../../../package.json` is missing from the local GSD install.
- **Fix:** Continued by following the checked-in PLAN directly and manually updating summary/state/roadmap artifacts.
- **Files modified:** Planning close-out files only
- **Verification:** Plan-level checks were run manually and recorded above.
- **Committed in:** plan metadata commit.

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** No scope creep. Both fixes preserve the plan outcome and keep the repo relayable across agents.

## Issues Encountered

- `uv` is installed as a user Python module. In this PowerShell session, `uv run` works after adding `%APPDATA%\Python\Python311\Scripts` to `PATH`; `py -m uv run ...` is also reliable.
- Local `gsd-tools.cjs` is currently broken, so subsequent GSD state updates may need to be manual unless the GSD install is repaired.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `01-02`: implement `BPETokenizer.train`, `encode`, and `decode` so `code/align_check.py --only vp1,vp2,vp4,vp6` turns green.

## Self-Check: PASSED

- Key files exist on disk.
- Task commits exist for tracked task outcomes.
- All plan-level verification commands passed, with the harness red state matching the plan's expected failure.

---
*Phase: 01-gates-first-submittable-tokenizer*
*Completed: 2026-06-25*
