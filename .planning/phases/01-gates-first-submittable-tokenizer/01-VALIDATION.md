---
phase: 1
slug: gates-first-submittable-tokenizer
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-25
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `01-RESEARCH.md` § Validation Architecture (VP-1..VP-8).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | None present — no pytest/unittest in the scaffold. Validation is via (a) our own assertion harness `code/align_check.py`, (b) provided `code/test_tokenizer.py`, (c) unedited `check_submission.py`, (d) `code/train_ner_model.py`. Plain `assert`s suffice — do NOT add a test framework (stack is locked to numpy/regex/torch/tqdm). |
| **Config file** | none — see Wave 0 (env bootstrap only) |
| **Quick run command** | `py code/align_check.py` (CPU; gates + alignment + pickle round-trip) — or `uv run python code/align_check.py` once env synced |
| **Full suite command** | `uv run python check_submission.py HW2_214034621.zip` (full gate incl. 1-batch NER smoke) |
| **Estimated runtime** | align_check ~seconds on a subsample; checker smoke ~1–2 min CPU |

---

## Sampling Rate

- **After every task commit:** Run `py code/align_check.py` (covers VP-1, VP-2, VP-3, VP-4, VP-6) — must be green.
- **After every plan wave:** Run `uv run python check_submission.py HW2_214034621.zip` (VP-8) on a freshly built zip; run the VP-5 double-run + `cmp` reproducibility diff.
- **Before `/gsd-verify-work`:** Full suite green AND VP-7 (F1 ≥ 0.4) shows a non-trivial directional F1 on a local CPU smoke of the identical NER pipeline.
- **Max feedback latency:** ~10 s for `align_check.py` on the seeded subsample.

---

## Per-Task Verification Map

> Task IDs are assigned by the planner/executor; rows below are the Validation Points (VPs) every relevant task must keep green. "Local" = CPU, no VM. Only the **scored** F1 magnitude (VP-7) needs the M60.

| VP | Requirement | Wave | Behavior | Test Type | Automated Command | Local CPU? | Status |
|----|-------------|------|----------|-----------|-------------------|------------|--------|
| VP-1 | CORE-02/03/04 | by-plan | ≥1 bigram token AND no token with 2+ internal spaces (mirror checker predicate `tok.replace('▁',' ').strip()`) | unit assertion | `py code/align_check.py` | ✅ Local | ⬜ pending |
| VP-2 | CORE-02 / SERDE-01 | by-plan | `space_token` non-None and present in `token_to_id` | unit assertion | `py code/align_check.py` | ✅ Local | ⬜ pending |
| VP-3 | SERDE-01/02 | by-plan | Pickle reload from `code/` alone in a FRESH process round-trips encode/decode | subprocess harness | `py code/align_check.py` (spawns fresh `py` on copied `code/`) | ✅ Local | ⬜ pending |
| VP-4 | ALIGN-01/02/03 | by-plan | Decode length-faithful: cumulative `decode(ids[:i+1])` monotonic + exact char offsets on dev_1/dev_2 (incl. non-ASCII & double-space lines); `decode(encode(x)).replace(' ','')==x.replace(' ','')` | local harness (mirrors `_get_token_char_spans_fast`) | `py code/align_check.py` | ✅ Local | ⬜ pending |
| VP-5 | SERDE-03 | by-plan | Byte-reproducibility: unedited `generate_tokenizers.py` twice → byte-identical `tokenizer_1.pkl` | double-run + diff | run script twice to two dirs; `cmp tokenizer_1.pkl` | ✅ Local (needs env) | ⬜ pending |
| VP-6 | CORE-05 | by-plan | Zero `[UNK]`: byte fallback → every char maps to in-vocab tokens (non-ASCII / emoji / OOV stress string never yields id 1) | unit assertion | `py code/align_check.py` | ✅ Local | ⬜ pending |
| VP-7 | EVAL-01 | by-plan | NER **F1 ≥ 0.4** on `dev_1_binary.tagged` via tokenizer 1, end-to-end | full NER pipeline | `uv run python code/train_ner_model.py --tokenizer_path trained_tokenizers/tokenizer_1.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged` | ⚠️ CPU smoke local; **scored number on M60 VM** | ⬜ pending |
| VP-8 | PKG-01 | by-plan | Full submission gate: zip structure + all 3 pkls load + NER smoke pass | checker run on built zip | `uv run python check_submission.py HW2_214034621.zip` | ✅ Local (CPU; smoke is 1 batch) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `code/align_check.py` — new OURS harness covering VP-1, VP-2, VP-3, VP-4, VP-6 (the local gate). Does not exist yet.
- [ ] Fresh-process pickle-load subprocess helper (may live inside `align_check.py`) — VP-3.
- [ ] A tiny **seeded** subsample path inside `BPETokenizer.train()` so the harness/check loop runs in seconds (D-03/D-04) — the cap must live in `train()` because `generate_tokenizers.py` reads the whole file and passes only `vocab_size`.
- [ ] Env bootstrap: `uv` not installed and no `.venv` — run `bash init.sh` (or `uv sync`) before any `uv run` / NER / checker step.
- [ ] No test-framework install (no pytest) — plain `assert`s only; adding libraries violates the locked stack.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Scored F1 magnitude on GPU | EVAL-01 (floor) / EVAL-02+ (later) | BiLSTM NER training is slow on CPU; the *scored* number is produced on the Tesla M60 VM (protect 10h budget) | Local CPU smoke must first show the pipeline runs end-to-end with directional F1; the confirmed scored run is dispatched to Codex-on-VM via GitHub. |

*Everything else has automated local verification — no gate is invisible until the VM.*

---

## Validation Sign-Off

- [ ] All tasks have an `<automated>` verify (align_check / checker / NER) or a Wave 0 dependency
- [ ] Sampling continuity: no 3 consecutive tasks without an automated verify
- [ ] Wave 0 covers all MISSING references (align_check.py, seeded subsample, env bootstrap)
- [ ] No watch-mode flags in any command
- [ ] Feedback latency < 10s for the quick command
- [ ] `nyquist_compliant: true` set in frontmatter (set by executor/auditor once the map is green)

**Approval:** pending
