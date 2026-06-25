# HANDOFF — Agent Relay Baton

> **Single source of truth for continuing this project across agents (Claude ⇄ Codex).**
> Read this FIRST. Update it LAST (before you stop). COMMIT it. The git history is our shared memory.

## Relay protocol (ping-pong)

1. **On start:** if a git remote exists, `git pull`. Then read THIS file top-to-bottom, especially **▶ NEXT ACTION**, then open the files it points to.
2. **Work the next chunk.** Commit early and often with clear, conventional messages (`feat(...)`, `fix(...)`, `docs(...)`). Small commits = the other agent can see exactly what changed.
3. **Before you stop / hit your limit:** update the sections below (**Current status**, **What was just done**, **▶ NEXT ACTION**, **Open questions**). Leave the tree clean — or, if work is mid-flight, write a `WIP:` note here saying exactly what's half-done. Then `git add -A && git commit`, and `git push` if a remote exists.
4. **Hand back the baton:** the user pastes the *other* agent its continuation prompt (bottom of this file).

## Project (one-liner)

From-scratch **BPE tokenizer** (`code/bpe_tokenizer.py`, `BPETokenizer(BaseTokenizer)`) feeding a **FIXED** binary NER model (word-level F1). Deliver **3 trained tokenizers + a ≤1-page report** as `HW2_214034621.zip`. Graded NLP homework (HW2). **NOT** a retrieval/search/ranking system — there is no candidate generation, reranking, or NDCG.

## Current status

- **Last updated by:** Claude — 2026-06-25 (at commit `694f9ea` + this HANDOFF commit)
- **Phase:** 1 of 6 — *Gates-First Submittable Tokenizer*
- **Phase status:** **PLANNED** — 4 executable plans, verified, 0 blockers. **Nothing implemented yet.**
- **Code state:** `code/` and `data/` are still inside the two root zips (not extracted). `uv` is not installed; no `.venv`.

## What was just done (most recent session — Claude)

- Ran `/gsd-plan-phase 1`: researched the phase, wrote the Nyquist `01-VALIDATION.md` (VP-1..VP-8), produced + verified 4 plans, resolved 2 checker warnings.
- Commits: `bb763ec` research → `608842d` validation → `d5d46bb` plans → `934e03d` finalize → `694f9ea` clean tree.

## ▶ NEXT ACTION

Execute Phase 1 **in order** (waves are strictly sequential 0→1→2→3 — they share `code/bpe_tokenizer.py`):

1. **`01-01`** — `.planning/phases/01-gates-first-submittable-tokenizer/01-01-PLAN.md`
   Extract `code_env_example.zip` + `data (3).zip` into the repo; `bash init.sh` (or `uv sync`); create `code/align_check.py` (the local gate harness, initially failing) + a SHA-256 manifest of the 4 forbidden files.
2. **`01-02`** — char-up BPE `train`/`encode`/`decode`: `▁` (U+2581) space marker, guaranteed cross-word bigram capped at one internal space, UTF-8 byte fallback (zero `[UNK]`), pure length-faithful decode.
3. **`01-03`** — serialization hardening (`__getstate__`/`__setstate__`, regex-as-string, cache drop) + deterministic merges for byte-reproducibility.
4. **`01-04`** — generate all 3 reproducible pkls, **local CPU** NER F1 smoke (directional F1≥0.4), build `HW2_214034621.zip` + `train_commands.txt`, pass full `check_submission.py`.

Read each PLAN.md's `<read_first>` / `<action>` / `<acceptance_criteria>` / `<verify>`. Run `code/align_check.py` after each step.

## Hard rules (violating any = wasted work or a zero grade)

- **Stack:** `numpy`, `regex` (`import regex`, never stdlib `re`), `torch`, `tqdm` ONLY. No other libraries.
- **NEVER edit** `base_tokenizer.py`, `train_ner_model.py`, `generate_tokenizers.py`, `check_submission.py` — they must stay byte-identical to the originals.
- **Pickle-safety:** no lambdas / closures / `defaultdict(lambda)` / local classes in tokenizer state; store regex as a pattern string; drop the encode cache in `__getstate__`/`__setstate__`.
- **decode** = pure length-preserving `▁`→single-space substitution (no stripping / normalization).
- **GPU (scored NER F1)** runs only on the M60 VM via Codex; keep local CPU gates green first to protect the 10h budget.

## Open questions / decisions needed from the user

- **Git remote:** none configured. Needed only for cross-machine relay (Codex on the VM). The data zip is ~95MB → it should be `.gitignore`d and kept on the VM separately; only `.planning/`, `code/`, and `HANDOFF.md` travel through git. *(Local Codex in this same folder needs no remote.)*
- Nothing else blocks executing `01-01` locally.

## Key files map

- Rules / constraints: `.claude/CLAUDE.md`
- Spec & roadmap: `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, `.planning/STATE.md`
- Research: `.planning/research/{STACK,ARCHITECTURE,PITFALLS,FEATURES,SUMMARY}.md`
- Phase 1: `.planning/phases/01-gates-first-submittable-tokenizer/{01-CONTEXT,01-RESEARCH,01-VALIDATION,01-01..01-04-PLAN}.md`

---

## Continuation prompts (the user pastes these)

**→ To CODEX (when Claude hands off):**
```
Continue this project (it's a from-scratch BPE tokenizer for NER — NOT a retrieval/ranking system).
1. If you pulled from git, you have the latest. Read HANDOFF.md at the repo root — it's the baton: current status, the exact NEXT ACTION, hard rules, and the file map. Follow it.
2. Do the next chunk of work. Commit early and often with clear messages.
3. Before you stop or hit your limit: update HANDOFF.md (Current status, What was just done, NEXT ACTION, Open questions), commit everything, and push if a remote exists — so Claude resumes exactly where you stopped.
Never edit the 4 forbidden files. If unsure about anything, say so instead of guessing. Ask before any irreversible or out-of-scope change.
```

**→ To CLAUDE (when Codex hands off):** just say:
```
Read HANDOFF.md and continue from the NEXT ACTION.
```
