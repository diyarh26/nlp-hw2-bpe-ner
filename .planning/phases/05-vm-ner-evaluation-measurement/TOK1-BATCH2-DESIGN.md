# tok1 Batch 2 — capitalization-aware + position-consistent pre-tokenization (DESIGN)

**Status:** designed, NOT implemented. Contingent: implement only if the Batch 1.5
mix-vocab sweep plateaus below the ~0.50 Twitter-F1 target.
**Author:** Claude — 2026-06-27 (designed in parallel with the Batch 1.5 sweep).

## Why
tok1 (Twitter) F1 is the weakest axis. Batch 1 found the balanced-mix DATA recipe
is best (0.4780 @ v6000). Batch 1.5 sweeps mix vocab (cheap, no code risk). If that
plateaus < ~0.50, the next lever is structural: change *how text is split into
pre-token chunks* before BPE, so subwords carry more NER-relevant signal.

## HARD CONSTRAINT (the whole design pivots on this)
The fixed NER pipeline (`code/train_ner_model.py`) has **no `encode_with_offsets`**,
so it derives every token's char span from **cumulative decode lengths**:
`expected_start = len(decode(ids[:i]))`, `token_end = len(decode(ids[:i+1]))`, then
clamps to `len(text)` and aligns tokens→words by char overlap (words = `text.split()`),
labeling the **first subtoken** of each word.

⇒ `decode` MUST stay length-faithful: `len(decode(ids[:k]))` equals the original
char offset after k tokens. This makes two "obvious" ideas **decode-UNSAFE** (they
shift every offset and break alignment / reconstruction — ALIGN-01/02):
- ❌ Capitalization **marker tokens** (`<cap>`) — decode to 0 or extra chars.
- ❌ Case-folding (`Apple`→`apple` + case flag) — decode can't restore `A` length-faithfully.
- ❌ Prepending `▁` to the **first** word for uniform word-initial marking — adds a
  leading space in decode → all offsets off by one.

## What IS decode-safe (the actual lever)
`decode` is just `token surface → replace ▁ with ' '` + byte-defallback; it is
**independent of how chunks were split**. So we can freely change the
**pre-tokenization regex** (which currently is effectively `\X` / per-char — see
`_text_to_train_symbols`, `_text_to_encode_symbols`; `pretoken_pattern`/
`_pretoken_regex` already exist and pickle as a string but are NOT wired into the
hot path). A structured regex changes which merges are *possible/learned* (the
vocab), not decode.

### Design: structured, Twitter-aware pre-tokenization
Split each line into pre-token chunks on linguistically/entity-meaningful
boundaries, each chunk still carrying its leading `▁` space marker so cross-space
**bigram** merges remain legal (≥1 bigram gate; ≤1 internal space cap unchanged):
- letters run | digits run | each punctuation/symbol | whitespace→`▁`
- keep Twitter artifacts as coherent units where it helps: `@mention`, `#hashtag`,
  URLs, emoji (emoji already isolate as their own grapheme/byte units).
- **Case transition as a boundary** is the decode-safe face of "capitalization-aware":
  e.g. split `lowerUPPER` and keep a leading capital letter as a chunk start, so a
  capitalized (proper-noun) word's **first subtoken is consistent** regardless of
  the rest of the word — without any marker token. The capital letter stays in the
  surface (decode-faithful); we only constrain where merges may begin.
- **Position consistency** comes for free: every word-internal chunk boundary is
  uniform; the only residual is the first word lacking a leading `▁` (intrinsic and
  decode-required — do NOT "fix" it).

## Implementation sketch (all in `code/bpe_tokenizer.py`, the editable core)
1. Set `PRETOKEN_PATTERN` to a `regex` pattern (string, pickle-safe) capturing the
   classes above; compile in `_init_runtime_state` (already done).
2. Rewrite `_text_to_chunks` (train) and `_text_to_encode_symbols` (encode) to:
   `for m in self._pretoken_regex.finditer(text): emit ▁ for spaces, else the chunk's
   chars, with byte-fallback for chars not in vocab`. Keep chunk = list-of-symbols.
3. Leave `decode`, `_decode_surface`, byte fallback, merge loop, heap/index, and
   `__getstate__/__setstate__` UNTOUCHED.

## Gates that MUST stay green (graded; any miss = risk of 0)
- `speed_check.py --only all` — encode equality oracle (slow vs fast) **and**
  train==full-recount. Both recompute with the SAME new pre-tok, so equality holds.
- `align_check.py` — length-faithful decode + first-subtoken mapping.
- `check_submission.py` — ≥1 bigram, 0 trigram (≤1 internal space), valid space_token,
  pickle-reload from `code/`, zip structure; forbidden files byte-identical.
- Zero `[UNK]` (byte fallback intact); deterministic / byte-reproducible training.

## Validation plan
1. Implement → run `speed_check.py --only all` + `align_check.py` locally (CPU).
2. Regenerate the tok1 candidates (best data recipe from Batch 1/1.5 × a small vocab
   grid) under the NEW pre-tok; CPU-measure efficiency + safety gates.
3. Vendor + push; VM F1 sweep on train_1/dev_1 (same harness as Batch 1).
4. Compare to the Batch 1.5 best; stage the global winner via the stock generator
   (byte-identity check) and rebuild the zip.

## Risk / fallback
Editing the core tokenizer risks the gates above and could *hurt* F1 (over/under-
segmentation). If Batch 2 underperforms the Batch 1.5 best, **revert** to the
pure-`\X` tokenizer and ship the Batch 1.5 winner — the pre-tok change is isolated
to `PRETOKEN_PATTERN` + the two `_text_to_*` methods, so revert is clean.
