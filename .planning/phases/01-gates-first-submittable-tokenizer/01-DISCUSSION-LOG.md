# Phase 1: Gates-First Submittable Tokenizer - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-24
**Phase:** 1-Gates-First Submittable Tokenizer
**Areas discussed:** Base alphabet, Phase-1 build scope, Starting vocab size (Whitespace/decode edges deferred to Claude)

---

## Gray-area selection

| Option | Description | Selected |
|--------|-------------|----------|
| Base alphabet (byte vs char) | byte-level vs char-level base + UNK strategy | ✓ |
| Phase-1 build scope | minimal/subsample vs efficient trainer now | ✓ |
| Starting vocab size | Phase-1 default vocab | ✓ |
| Whitespace/decode edges | multi-space/tab/newline + marker handling | (deferred to Claude) |

---

## Base alphabet

| Option | Description | Selected |
|--------|-------------|----------|
| Char-level + byte fallback | Char base; unseen chars → UTF-8 byte tokens → zero UNK; exact length-faithful decode on domains 1&2 | ✓ |
| Byte-level base (GPT-2 style) | 256-byte base; maximal robustness but trickier multibyte decode/alignment | |
| You decide | Claude picks for most-robust alignment | |

**User's choice:** Char-level + byte fallback
**Notes:** Keeps the F1-critical decode aligner exact on the measured domains while still guaranteeing zero UNK on the hidden domain via the byte-fallback layer.

---

## Phase-1 build scope

| Option | Description | Selected |
|--------|-------------|----------|
| Subsample for fast signal | Train Phase 1 on a capped subset; defer full-corpus + efficiency to Phase 2 | ✓ |
| Build efficient trainer now | Implement incremental/indexed trainer in Phase 1 to handle full 95MB | |
| You decide | Choose based on naive-trainer speed | |

**User's choice:** Subsample for fast signal
**Notes:** MVP-aligned — get gates + F1≥0.4 signal fast; efficient full-corpus training lands in Phase 2 before final reproducible tokenizers. Subsample must be seeded/deterministic and live inside `train()` since `generate_tokenizers.py` is unedited.

---

## Starting vocab size

| Option | Description | Selected |
|--------|-------------|----------|
| ~8000 | Closer to research F1 sweet spot (~8k–16k), still fast, clears 0.4 | ✓ |
| 5000 (scaffold default) | Leaner/faster, easily clears 0.4 | |
| You decide | Sane default; tuned in Phase 3 | |

**User's choice:** ~8000
**Notes:** Starting point only; Phase 3 sweeps vocab size per domain.

---

## Claude's Discretion

- Whitespace / decode edge handling (multi-space, tabs, newlines, leading/trailing): preserve exact characters, one `▁` per literal space, no length-changing normalization; confirm empirically against dev data.
- Pre-tokenization regex pattern (GPT-2-like, space-aware, Twitter-tolerant), keeping `▁` an emitted mergeable symbol.
- Merge tie-breaking rule for deterministic reproducibility (e.g. lexicographic on the pair).
- Exact Phase-1 subsample size and the bigram-forcing threshold.

## Deferred Ideas

- Encode-speed / training-efficiency (rank-map encode, per-chunk cache, incremental inverted-index training, full 95MB) → Phase 2.
- Per-domain vocab sweep & tuning → Phase 3.
- Tokenizer-3 data-mix / dedup / generalization → Phase 4.
- Twitter-specific preprocessing (URLs/@/#/emoji) → v2 OPT-03 if it helps without breaking decode faithfulness.
