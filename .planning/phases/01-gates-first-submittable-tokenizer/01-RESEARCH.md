# Phase 1: Gates-First Submittable Tokenizer - Research

**Researched:** 2026-06-25
**Domain:** From-scratch BPE tokenizer (`code/bpe_tokenizer.py`) plugged into a FIXED course NER scaffold; Phase-1 scope = pass every hard disqualification gate, decode length-faithfully, pickle/reproduce from `code/`, clear F1≥0.4 on domain 1.
**Confidence:** HIGH (every contract below was verified against the **actual extracted scaffold source**, not inferred from PROJECT.md; the load-bearing alignment behavior was empirically simulated end-to-end)

## Summary

This phase is an engineering problem against a fixed, graded harness, not a research problem. The single most valuable thing this research did was **extract and read the real scaffold** (`check_submission.py`, `base_tokenizer.py`, `train_ner_model.py`, `generate_tokenizers.py`, `test_tokenizer.py`, `train_tokenizer.py`, sample `HW2_123456789.zip`) and **empirically simulate the NER aligner**. That turned several "MEDIUM, assignment-specific" claims from the project-level research into HIGH-confidence, verified contracts, and surfaced concrete planner unknowns that the prose research did not pin down (exact bigram predicate, exact char-span algorithm, exact zip layout, exact load mechanism, in-`train()` subsample requirement).

The decode/alignment contract is the linchpin and it is now **verified**: `train_ner_model.py`'s `NERDataset._get_token_char_spans_fast` builds token char-spans purely from `len(decode(ids[:i+1]))` deltas, clamped to `len(text)`, then assigns each word to the token with maximum char-span overlap and labels the *first* such token. A pure length-preserving decode (`''.join(id_to_token[i] for i in ids).replace('▁',' ')`) makes this alignment exact — I simulated the README's own "Steve Jobs" example through the real algorithm and every word mapped to the correct first subtoken, including the cross-word-bigram case, with standalone space tokens correctly absorbed (`sub_to_word = -1`).

**Primary recommendation:** Build a *correct, simple* char-up BPE with a mergeable `▁` (U+2581) marker, a hard ≤1-internal-space merge cap, deterministic tie-breaking (lexicographic), a 256-byte fallback alphabet for zero-UNK, and a pure length-preserving decode. Put the Phase-1 training cap **inside `train()`** (seeded line cap) because the unedited `generate_tokenizers.py` reads the whole file and only passes `vocab_size`. Front-load all gates to local CPU; the only thing that needs the M60 VM is the final F1≥0.4 run on `dev_1_binary.tagged` — and even that runs (slowly) on CPU for a smoke pass.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Base alphabet is **char-level** — characters observed in the (subsampled) training data become atomic base tokens, **plus all 256 single-byte values** to back the fallback.
- **D-02:** **Zero UNK via UTF-8 byte fallback** — any character not in the char vocab is decomposed into its UTF-8 byte tokens (always in vocab). `[UNK]` is never emitted by `encode`. Keeps the length-faithful decode exact on domains 1 & 2 while degrading gracefully on the hidden domain.
- **D-03:** **Subsample `domain_1_train.txt` for Phase 1** — train the correct-but-simple BPE on a capped subset so a naive trainer runs in seconds–minutes and yields gates + F1≥0.4 fast. The full-corpus efficient trainer is Phase 2.
- **D-04:** The subsample must be **deterministic/seeded** (SERDE-03). The Phase-1 cap must live **inside `BPETokenizer.train()`** (seeded line cap / constant) or be exercised via `code/train_tokenizer.py` on a pre-subsampled file — **never** by editing `generate_tokenizers.py`.
- **D-05:** Phase-1 starting `vocab_size ≈ 8000` (starting point only; Phase 3 sweeps per domain — Phase 1 must not hard-code assumptions that block a sweep).
- **D-06:** Space marker is `▁` (U+2581), added to vocab as a **first-class mergeable symbol**; `space_token` attribute set to it.
- **D-07:** Merges may cross the marker but are **hard-capped at one internal space** (filter any candidate whose merged surface has ≥2 markers, in BOTH train and encode). Guarantee ≥1 bigram deterministically: if none emerges, **force the highest-count legal cross-space merge**.
- **D-08:** Validate the bigram gate the assignment-specific way: a token is a bigram iff `tok.replace('▁',' ').strip()` contains an internal space; assert ≥1 such token and **no** token with 2+ internal spaces. (Mirror `check_submission.py` exactly.)
- **D-09:** `decode` is a pure, length-preserving substitution: ids→tokens→concatenate→replace `▁` with a single space. **No** stripping, case-folding, NFC/normalization, or any transform that changes character count. Special tokens decode in a way that does not corrupt cumulative length.
- **D-10:** Build a **local alignment harness** (own script under `code/`) asserting, on dev_1/dev_2 samples: cumulative `decode(ids[:i+1])` lengths are monotonic and reconstruct exact char offsets, and `decode(encode(x)).replace(' ','') == x.replace(' ','')`. Green locally before any VM run.
- **D-11:** All learned state in plain picklable containers (`dict`/`list`/`tuple`/`str`/`int`): merge ranks `{(str,str):int}`, vocab maps, `space_token`. **No** lambdas/closures/`defaultdict(lambda)`/local classes. Convert any `defaultdict` to plain `dict` before pickling.
- **D-12:** Store the regex as a **pattern string** (compile in `__init__`/`__setstate__`); drop the runtime encode cache via `__getstate__`/`__setstate__` and rebuild on load. Class stays in `code/bpe_tokenizer.py` so it unpickles from `code/` alone.

### Claude's Discretion
- **Whitespace / decode edges:** preserve exact characters; emit one `▁` per literal space; decide multi-space/tab/newline handling empirically from dev data so encode/decode stays length-symmetric (the measured NER `text` is single-space-joined via `' '.join(words)`).
- **Pre-tokenization regex** pattern (GPT-2-like, space-aware, Twitter-tolerant) — exact pattern is mine; must keep the marker as a real emitted symbol so cross-space merges stay legal.
- **Merge tie-breaking rule** for deterministic reproducibility — pick a deterministic rule (e.g. lexicographic on the pair) and document it.
- **Exact Phase-1 subsample size** and the **bigram-forcing threshold** — my discretion, tuned to keep local iteration fast while clearing F1≥0.4.

### Deferred Ideas (OUT OF SCOPE)
- **Encode-speed / training-efficiency** (rank-map greedy + per-chunk cache + incremental inverted-index training on the full 95MB) → Phase 2 (SPEED-01..03).
- **Per-domain vocab sweep & tuning** for tokenizers 1 & 2 → Phase 3 (TUNE-01..04).
- **Tokenizer-3 data-mix / dedup / generalization** → Phase 4 (GEN-01..03).
- **Twitter-specific preprocessing** (URLs/@/#/emoji) as an efficiency/F1 lever → v2 OPT-03.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CORE-01 | True char-up BPE, merge most-frequent adjacent pair until `vocab_size` | Verified `generate_tokenizers.py` calls `BPETokenizer(vocab_size=...)` then `.train(texts)`; stub `__init__(self, vocab_size=10000)`. Subsample cap must live in `train()` (D-04). Standard Stack §Core + Architecture Pattern 1. |
| CORE-02 | `space_token='▁'` first-class mergeable symbol, never None | Verified `check_submission.check_tokenizer_requirements` rejects `space_token is None`; `NERDataset` reads `tokenizer.space_token in token_to_id`. Marker has **zero collision** in train data (verified). |
| CORE-03 | Merges cross the marker → ≥1 bigram token | Verified bigram predicate (Code Examples §Bigram gate). Force-merge fallback (D-07) is disqualification insurance. |
| CORE-04 | Hard-cap at one internal space (no trigram+) | Eligibility filter `(a+b).count('▁') <= 1` in train AND encode (Pattern 2). |
| CORE-05 | Byte/char base alphabet → zero UNK | dev data **contains non-ASCII** (Arabic U+0650, smart quotes — verified), so byte fallback is needed even on measured domains, not just hidden. 256-byte alphabet guarantees coverage. |
| ALIGN-01 | `decode` restores spaces, length-faithful | Verified char-span algorithm in `_get_token_char_spans_fast` (Code Examples). Pure `.replace('▁',' ')` decode is exact-length. |
| ALIGN-02 | `decode(encode(text))` reconstructs text per checker | Verified `test_reconstruction` uses `reconstructed.replace(' ','') == text.replace(' ','')` — **whitespace-insensitive** (so the byte-fallback edge is forgiven on spaces only). |
| ALIGN-03 | Local alignment harness before VM | Simulated the real aligner end-to-end (Validation Architecture VP-4). Harness is a new `code/` helper (D-10). |
| SERDE-01 | Pickle/unpickle from `code/` alone, plain state | Verified `BaseTokenizer.save/load` = bare `pickle.dump/load(self)`; `check_submission` copies `code/*.py` to root then `BaseTokenizer.load`. Class must stay importable as `bpe_tokenizer.BPETokenizer`. |
| SERDE-02 | Regex as pattern string; drop cache via `__getstate__/__setstate__` | D-12. Verified no state survives except what we pickle. |
| SERDE-03 | Byte-reproducible via unedited `generate_tokenizers.py` (seed=42) | Verified script seeds `random`+`numpy` only (not torch). Determinism is OUR responsibility: total tie-break + seeded in-`train()` subsample (D-04). |
| EVAL-01 | NER F1 ≥ 0.4 on `dev_1_binary.tagged` via tokenizer 1 | Verified fixed NER HP (BATCH=32, LR=0.01, EPOCHS=20, SEED=42, BiLSTM). Run end-to-end (CPU smoke locally, M60 for the scored number). |
| PKG-01 | Zip matches structure, passes `check_submission.py` | Verified exact required layout from sample zip + checker constants (Code Examples §Zip layout). |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| BPE training (merges/vocab) | `BPETokenizer.train()` | — | Only `vocab_size` passed by harness; subsample cap MUST live here (D-04) since `generate_tokenizers.py` reads whole file. |
| Encode (text→ids) | `BPETokenizer.encode()` | per-chunk cache (runtime, not pickled) | Pre-tokenize→symbols→apply merges. Cache dropped on pickle (D-12). |
| Decode (ids→text) | `BPETokenizer.decode()` | — | Pure length-preserving substitution; the load-bearing NER-alignment contract lives entirely here (D-09). |
| Token↔word alignment | **FIXED scaffold** (`NERDataset`) | our decode-length faithfulness | We do NOT own alignment; we own the decode-length invariant it consumes. |
| Persistence | **FIXED** `BaseTokenizer.save/load` (pickle) | our picklable state (D-11/D-12) | Inherit unchanged; only guarantee state is plain + class path stable. |
| Reproducibility | unedited `generate_tokenizers.py` | our determinism (tie-break + seeded subsample) | Script seeds RNGs; our code must be order-deterministic. |
| Gate validation | FIXED `check_submission.py` | our local mirror of its predicates | Mirror checker predicates locally so we never discover a DQ in the final build. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `regex` (3rd-party) | `>=2026.5.9` (env-pinned) | Pre-tokenization into space-marked chunks; `re.escape`/`re.search` mirrors what `train_ner_model.py` itself uses | Only allowed advanced regex engine; `train_ner_model.py` imports `regex as re` — same engine. Store the **pattern string**, compile lazily (D-12). [CITED: code/train_ner_model.py:12] |
| `numpy` | `>=2.4.6` (env-pinned) | Optional bulk pair counting; keep *learned state* as Python primitives, not arrays (pickle portability) | Allowed; `generate_tokenizers.py` seeds it. [CITED: pyproject.toml] |
| `pickle` (stdlib) | default protocol | Persistence (inherited via `BaseTokenizer.save/load`) | Verified: `pickle.dump(self)` / `pickle.load`. [CITED: code/base_tokenizer.py:108-123] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `collections.Counter` | stdlib | Seed initial pair counts once | Phase-1 simple trainer; do NOT recount every merge (that's the anti-pattern). |
| `tqdm` | `>=4.68.1` (env-pinned) | Progress bars over merge loop / corpus pass | Dev ergonomics; `train_ner_model.py` already uses it. [CITED: pyproject.toml] |
| `torch` | `>=2.12.0`, **cu126 index** | Used by the FIXED NER pipeline only — we do not call it | Pulled from PyTorch cu126 index for M60 (Maxwell sm_52); default cu13 wheel silently falls back to CPU. [CITED: pyproject.toml comments] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Simple correct O(n·merges) encode | Linked-list + rank-heap O(n log n) | Speed is **Phase 2** (deferred). Phase 1 uses the simple correct path as the equality oracle. |
| In-`train()` seeded line cap | Pre-subsampled file via `train_tokenizer.py` | Both valid for *local dev*; but the **reproducible** Phase-1 pkl from unedited `generate_tokenizers.py` requires the cap **inside `train()`** (D-04) — the script passes only `vocab_size`. |
| 256-byte fallback alphabet | Full byte-level BPE | Byte-level complicates the length-faithful aligner on measured domains; char-up + byte fallback keeps decode exact where F1 is scored (D-02 rationale). |

**Installation:** No new packages. The env is fully specified by the provided `pyproject.toml` + `uv.lock`; `bash init.sh` runs `uv sync`. **Do not add dependencies** (spec violation = non-reproducible).

**Version verification:** All four libraries (numpy, regex, torch, tqdm) are **env-pinned in the provided `pyproject.toml`** and locked in `uv.lock`. We add none, so no registry lookup is required or permitted.

## Package Legitimacy Audit

**Not applicable to Phase 1.** This phase installs **zero** external packages. The dependency set (numpy, regex, torch, tqdm) is fixed by the course-provided `pyproject.toml` + `uv.lock` and adding any package is an explicit spec violation (Out of Scope, REQUIREMENTS.md). No package legitimacy gate runs because the planner must not introduce installs.

- **Packages removed due to [SLOP] verdict:** none (none considered).
- **Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
raw line (one sentence)                          tokenizer_1.pkl (pickle of self)
      │                                                  ▲
      │ train(texts)  [SEEDED SUBSAMPLE CAP HERE]        │ BaseTokenizer.save (inherited)
      ▼                                                  │
 pre-tokenize → space-marked symbol stream         learned state (D-11):
      │  (literal space → '▁', emitted as symbol)    token_to_id / id_to_token (ids ≥ 4)
      ▼                                               merges list + rank dict {(a,b):rank}
 count adjacent pairs (Counter, once)                 space_token='▁', vocab_size, regex-string
      │                                                  │
      ▼  greedy: pick max-count pair                     │ BaseTokenizer.load (fresh process, code/ only)
   ELIGIBILITY FILTER (a+b).count('▁') ≤ 1               ▼
   tie-break: lexicographic on (a,b)            ┌──────────────────────────────────────────┐
      │  record merge (rank), add merged token   │ FIXED NER pipeline (do not edit)          │
      ▼  repeat until len(vocab)==vocab_size      │ NERDataset:                               │
   FORCE ≥1 legal cross-space bigram if none      │  encode(text) → ids                       │
      │                                           │  spans[i] = (len(decode(ids[:i])),        │
      ▼                                           │              len(decode(ids[:i+1])))      │
   merges + vocab + space_token                   │              clamped to len(text)         │
                                                  │  word→token by max char-span overlap      │
 encode(text):                                    │  label FIRST overlapping token; rest=-100 │
   pre-tokenize → symbols                         │  BiLSTM → P/R/F1                          │
   apply merges greedily (lowest rank),           └──────────────────────────────────────────┘
   ELIGIBILITY FILTER again (belt & suspenders)
   per-chunk cache (runtime; dropped on pickle)
      │
      ▼ ids
 decode(ids): ''.join(id_to_token[i]) .replace('▁',' ')   ← PURE, length-preserving (D-09)
```

### Recommended Project Structure
```
NLP-HW2/
├── code/
│   ├── base_tokenizer.py        # FIXED — local only, NOT in zip
│   ├── bpe_tokenizer.py         # OURS — the entire tokenizer (the ONE submitted logic file)
│   ├── train_ner_model.py       # FIXED — local F1 run, NOT in zip
│   ├── test_tokenizer.py        # FIXED helper — MAY edit/extend, ships in zip
│   ├── train_tokenizer.py       # FIXED helper — MAY edit/extend, ships in zip
│   └── align_check.py           # OURS — local alignment + gate harness (D-10); ships in code/ (harmless)
├── generate_tokenizers.py       # FIXED (root) — produces 3 pkls, NOT in zip
├── check_submission.py          # FIXED (root) — final gate, NOT in zip
├── data/                        # provided corpora (NOT in zip)
├── trained_tokenizers/          # output: tokenizer_{1,2,3}.pkl  (IN zip)
├── train_commands.txt           # exact reproducible commands (IN zip root)
└── report_214034621.pdf         # ≤1-page report (IN zip; Phase 6 produces real one)
```

### Pattern 1: SentencePiece-style mergeable space marker (length-faithful decode)
**What:** Replace each literal space with `▁` (U+2581) as a real, mergeable vocab symbol; decode = `''.join(tokens).replace('▁',' ')`.
**When to use:** Mandatory — the NER aligner derives char spans from decode lengths.
**Example:**
```python
# Source: synthesized from verified train_ner_model.py contract + CONTEXT D-06/D-09
def decode(self, token_ids):
    pieces = [self.id_to_token[i] for i in token_ids
              if i not in (0, 2, 3)]            # drop PAD/BOS/EOS surface; keep them length-0
    return "".join(pieces).replace(self.space_token, " ")
```

### Pattern 2: Cross-word merges hard-capped at one internal space
**What:** Allow a merge to cross exactly one `▁`; forbid any merge whose result holds ≥2 markers. Apply in BOTH train (don't record the merge) and encode (skip it).
**Example:**
```python
# Source: CONTEXT D-07; mirrors check_submission bigram/trigram predicate
def _mergeable(self, a, b):
    return (a + b).count(self.space_token) <= 1   # ≤1 internal space → ≤ bigram
```

### Pattern 3: Deterministic greedy merge selection (reproducibility)
**What:** When picking the max-count pair, break ties with a **total** deterministic key (lexicographic on the pair). Never rely on dict/`Counter.most_common`/set iteration order.
**Example:**
```python
# Source: PITFALLS Pitfall 6; CONTEXT discretion (tie-break rule)
best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))  # NOTE: lambda used at runtime only,
# NEVER stored on self — define as a module-level function if you prefer to avoid lambdas entirely.
```
> Caveat: the lambda above is fine *inside a method body* (never pickled). The pickle ban is on lambdas/closures stored as **instance state** (D-11).

### Anti-Patterns to Avoid
- **GPT-2 whitespace pre-tokenization** (never merges across spaces) → zero bigrams → DQ. (PITFALLS Pitfall 1.)
- **Trigram leakage** — merging a bigram token with an adjacent word → 2+ internal spaces → DQ. The eligibility filter (Pattern 2) is the twin guard. (Pitfall 2.)
- **Non-length-faithful decode** — `.strip()`, case-fold, NFC, whitespace collapse, or special-token surface leakage → silent F1 collapse. (Pitfall 3.)
- **Unpicklable state** — compiled `regex.Pattern`, lambda/closure/`defaultdict(lambda)`/local class on `self` → `BaseTokenizer.load` crash → DQ. (Pitfall 5.)
- **Editing `generate_tokenizers.py` to subsample** — the cap must live in `train()` (D-04); editing it = forbidden-file edit = grade 0.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Persistence / load | Custom serialize/deserialize | Inherited `BaseTokenizer.save/load` (bare pickle) | Verified the checker calls exactly this; a custom format won't be loaded. |
| Token↔word alignment | Your own char-span aligner | Just guarantee length-faithful decode; the FIXED `NERDataset` does the alignment | We don't own (and can't edit) the aligner. |
| NER training/eval | Any model code | FIXED `train_ner_model.py` (BiLSTM, fixed HP/seed) | Only file-path args may change; everything else is graded uniform. |
| Special-token ids | Re-numbering specials | Keep `[PAD]=0,[UNK]=1,[BOS]=2,[EOS]=3` from `BaseTokenizer.__init__`; learned tokens get ids ≥ 4 | Verified base `__init__` seeds these; collisions break decode. |
| Reproducible RNG seeding | Your own seeding in `train()` | `generate_tokenizers.py` already seeds `random`+`numpy` (seed=42) before `train()` | Don't re-seed; just be order-deterministic. |

**Key insight:** Phase 1 is "implement three methods inside a tight contract." Almost everything else is provided and graded-against-original — the surface area we own is `train`/`encode`/`decode` + state shape + a local harness. Maximize reuse, minimize footprint.

## Common Pitfalls

### Pitfall 1: [DQ] Subsample cap put in the wrong place breaks reproducibility
**What goes wrong:** You subsample by editing `generate_tokenizers.py` (forbidden edit → grade 0) or by pre-slicing a file that the grader won't have, so the regenerated pkl ≠ submitted pkl.
**Why it happens:** `generate_tokenizers.py` reads the *entire* file and passes only `vocab_size` — there's no CLI hook for a cap.
**How to avoid:** Put a **seeded, constant** line cap inside `BPETokenizer.train()` (D-04). The grader's unedited run then reproduces the same subset → same pkl.
**Warning signs:** Two runs of `generate_tokenizers.py` produce different `tokenizer_1.pkl`; cap logic lives outside `bpe_tokenizer.py`.

### Pitfall 2: [DQ/GATE] Decode not length-faithful → silent F1 collapse
**What goes wrong:** Any decode that changes char count (strip/normalize/collapse/special-token leakage) drifts the cumulative `len(decode(ids[:i+1]))` spans; labels land on wrong tokens; F1 silently tanks below 0.4.
**How to avoid:** Pure `''.join(...).replace('▁',' ')`; specials decode to length-0 surface. Verify with the cumulative-prefix harness (Validation VP-4) — **not** a naive round-trip.
**Warning signs:** Cumulative decode length non-monotonic or overshoots `len(text)`; F1 near the entity base rate despite a sane tokenizer.

### Pitfall 3: [DQ] No bigram / trigram leakage
**What goes wrong:** Zero bigram token (DQ) or a token with 2+ internal spaces (DQ).
**How to avoid:** Mergeable marker + eligibility filter in train AND encode + force ≥1 legal cross-space bigram if none emerged (D-07). Post-train, run the checker's own predicate over your vocab.
**Warning signs:** `check_submission` prints "No bigram token found" or a trigram; your own predicate scan returns no bigram or finds a 2+-space token.

### Pitfall 4: [DQ/REPRO] Pickle won't reload from `code/` alone
**What goes wrong:** Compiled regex / lambda / closure / `defaultdict(lambda)` / renamed-or-moved class on `self` → `BaseTokenizer.load` in a fresh process (cwd at the copied `code/`) crashes.
**How to avoid:** Store regex as a **string** and compile in `__init__`/`__setstate__`; drop the runtime cache via `__getstate__`; keep the class as `bpe_tokenizer.BPETokenizer`; convert any `defaultdict` to plain `dict` before pickling (D-11/D-12). After every edit, run the fresh-process load test.
**Warning signs:** `PicklingError` / `AttributeError: Can't get attribute` on load; works in-process but fails in a new interpreter.

### Pitfall 5: [REPRO] Non-deterministic tie-breaking
**What goes wrong:** Ties broken by dict/`Counter`/set iteration order → run-to-run vocab drift → regenerated pkl ≠ submitted pkl.
**How to avoid:** Total deterministic tie-break (lexicographic on the pair); sort before iterating where order matters. Verify by running `generate_tokenizers.py` twice and diffing pkls byte-for-byte.

### Pitfall 6: [GATE] Byte-fallback decode + whitespace edge
**What goes wrong:** A multibyte char split into byte tokens must decode back to the **same** characters. If your byte tokens are stored as surface strings that don't round-trip through UTF-8, decode length/chars drift on non-ASCII (and dev data **does** contain non-ASCII — verified).
**How to avoid:** Decide the byte-token representation so `decode` reconstructs the original bytes→chars exactly. Note the checker's reconstruction test is **whitespace-insensitive** (`.replace(' ','')`), so only *space* handling is forgiven — non-space chars must reconstruct exactly. Add non-ASCII dev lines to the alignment harness.

## Runtime State Inventory

> Phase 1 is greenfield (new `bpe_tokenizer.py` logic) — no rename/refactor/migration. This section is included only to record that the question was checked.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore; tokenizer state is self-contained in the pkl. | none |
| Live service config | None — fully offline coursework. | none |
| OS-registered state | None. | none |
| Secrets/env vars | None. | none |
| Build artifacts | The two project-root zips (`code_env_example.zip`, `data (3).zip`) are **not yet extracted into the repo**; extracting them into `code/` + `data/` is the first Phase-1 task. A read-only copy was extracted to the session scratchpad for this research only. | Extract zips as task 1; keep forbidden files byte-identical to originals. |

**Nothing found in stored-data / live-service / OS / secrets categories — verified by reading the scaffold (no DB, network, scheduler, or env usage anywhere).**

## Code Examples

Verified patterns from the **actual extracted scaffold** (paths are scratchpad copies of the originals inside the zips).

### Bigram / trigram gate predicate (mirror EXACTLY)
```python
# Source: check_submission.py:276-292 (verified)
special = getattr(tokenizer, 'special_tokens', {})
for tok in getattr(tokenizer, 'token_to_id', {}):
    if tok in special:
        continue
    surface = tok.replace(space, ' ') if space else tok   # space = tokenizer.space_token
    if ' ' in surface.strip():
        bigram_example = tok        # ≥1 such token REQUIRED, else DQ
        break
# Our local mirror must ALSO assert no token has surface.strip().count(' ') >= 2  (trigram → DQ)
```

### The exact NER char-span algorithm (the decode contract)
```python
# Source: train_ner_model.py:117-154  NERDataset._get_token_char_spans_fast (verified)
decoded_so_far = ""
for i, token_id in enumerate(token_ids):
    current_token_str = self.tokenizer.decode([token_id])      # per-token surface
    expected_start = len(decoded_so_far)
    decoded_so_far = self.tokenizer.decode(token_ids[:i + 1])   # cumulative
    token_start = expected_start
    token_end   = len(decoded_so_far)
    token_start = max(0, min(token_start, len(text)))           # clamped to text length
    token_end   = max(token_start, min(token_end, len(text)))
    token_char_spans.append((token_start, token_end))
# words then aligned by MAX char-span OVERLAP; FIRST overlapping token gets the label; rest = -100.
```
> Empirically simulated (this research) on the README "Steve Jobs ate an apple ." example with a length-faithful decode + cross-word bigram: every word mapped to its correct first subtoken; standalone `▁` tokens got `sub_to_word = -1` (absorbed). This is the proof that D-09 + Pattern 1 are sufficient. **Note `encode_with_offsets` shortcut:** `_get_token_char_spans_fast` first tries `self.tokenizer.encode_with_offsets(text)` if present — we do **not** define it, so the length-based Strategy 2 above is the path that runs. (Defining it later is a valid Phase-2/3 alignment-precision lever, out of scope here.)

### How the tokenizer is loaded at grade time
```python
# Source: check_submission.py:106-143, 295-311 (verified)
# 1. unzip submission; 2. chdir into it; 3. copy code/*.py to the extract-dir ROOT;
# 4. add extract-dir to sys.path; 5. tokenizer = BaseTokenizer.load(".../tokenizer_i.pkl")
#    => unpickle resolves the class as top-level module `bpe_tokenizer` (copied to root).
# Implication: class MUST be `bpe_tokenizer.BPETokenizer`; any helper the pickle references
# must be importable from a code/*.py file (which all get copied to root).
```

### Exact required zip layout (verified from sample HW2_123456789.zip + checker constants)
```
HW2_214034621.zip
├── report_214034621.pdf              # REQUIRED (checker errors if missing; Phase 6 supplies real PDF)
├── trained_tokenizers/
│   ├── tokenizer_1.pkl               # REQUIRED
│   ├── tokenizer_2.pkl               # REQUIRED
│   └── tokenizer_3.pkl               # REQUIRED
├── code/
│   └── bpe_tokenizer.py              # REQUIRED (only this is hard-required; helpers optional)
│   (sample also ships test_tokenizer.py, train_tokenizer.py — OK to include)
└── train_commands.txt                # SOFT WARNING if missing (not a structure failure) — include it
```
> Verified facts that change the plan: (a) `train_commands.txt` absence is **only a warning**, not a DQ (checker line 218-222) — but the README+spec require it, so include it. (b) The sample zip's `code/` does **NOT** contain `base_tokenizer.py` / `train_ner_model.py` — do not ship the forbidden files. (c) Filename ID must be `214034621` in `HW2_214034621.zip` and `report_214034621.pdf`. (d) `check_submission` runs the NER smoke (1 batch) for tokenizers 1&2 on domains 1&2 and tokenizer 3 on domain 1 — so **all three pkls must load and run** even in Phase 1 (use the default `generate_tokenizers.py` train-file mapping to produce all three).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| GPT-2 whitespace pre-tokenization (no cross-word merges) | SentencePiece `▁` (U+2581) mergeable marker + capped cross-word merges | This assignment's spec | Enables the required bigram; lossless `replace('▁',' ')` detokenization. [CITED: arXiv:1808.06226] |
| Naive `encode(decode(x))==x` round-trip test | Cumulative-prefix length-faithfulness test mirroring the aligner | Driven by `train_ner_model.py`'s length-based alignment (verified) | Catches silent F1 collapse the round-trip test misses. |

**Deprecated/outdated:**
- Default PyPI torch (CUDA 13) on this hardware: drops Maxwell → `torch.cuda.is_available()` False → NER silently runs on CPU. The provided `pyproject.toml` pins the **cu126** index for the M60. We don't touch torch, but the planner should know GPU availability depends on `bash init.sh` having synced the cu126 wheel. [CITED: pyproject.toml comments]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A simple correct char-up BPE on a seeded subsample of `domain_1_train.txt` at `vocab_size≈8000` clears F1≥0.4 on `dev_1_binary.tagged`. | EVAL-01 / D-03/D-05 | If F1 < 0.4, planner must widen the subsample or adjust vocab (still Phase-1 scope; tuning is Phase 3). The 0.4 floor is modest for a BiLSTM with 20 epochs, so risk is LOW but unverified until the run. |
| A2 | Lexicographic tie-break on the pair yields byte-identical pkls across two `generate_tokenizers.py` runs. | SERDE-03 / Pattern 3 | If pkl bytes differ (e.g. dict insertion-order effects on `id_to_token`), reproducibility fails. Mitigation: also assign ids in a deterministic, sorted/merge-order sequence. Verify by double-run diff. |
| A3 | A 256-byte fallback alphabet + char base covers all dev/hidden input with zero UNK and length-faithful decode of non-ASCII. | CORE-05 / Pitfall 6 | If byte-token surface representation doesn't round-trip UTF-8 exactly, non-ASCII dev lines misalign. Mitigation: include non-ASCII lines in the harness (dev data has Arabic/smart-quote chars — verified present). |
| A4 | Specials decoding to length-0 surface does not corrupt cumulative length; the aligner's `sub_to_word=-1` for marker/special tokens is harmless. | D-09 / Validation VP-4 | If a special leaks non-zero surface, spans drift. Mitigation: the simulation showed standalone markers get -1 cleanly; assert specials excluded in decode. |
| A5 | Phase-1 subsample size that "runs in seconds–minutes" naively still produces enough merges to guarantee a legal cross-space bigram (or the force-merge path fires). | D-07 / discretion | If subsample too small, few cross-space pairs; force-merge must reliably fire. Mitigation: make force-merge unconditional-fallback and test on the chosen subsample. |

## Open Questions

1. **Exact Phase-1 subsample size (lines/MB) and seeding mechanism inside `train()`.**
   - What we know: cap must be in `train()`, seeded; `generate_tokenizers.py` pre-seeds `random`+`numpy`(42) before calling `train()`.
   - What's unclear: whether to take the first-N lines (trivially deterministic, no RNG needed) or a seeded random sample. First-N is simplest and reproducible without touching RNG; a seeded sample needs care since the script already seeded globally.
   - Recommendation: **first-N lines** (e.g. N≈50k–200k of the 1.28M) — deterministic by construction, no RNG-order coupling, easiest to reproduce. Tune N up if F1<0.4 (A1).

2. **Byte-token surface representation for the fallback.**
   - What we know: must reconstruct exact bytes→chars in decode; dev data has non-ASCII.
   - What's unclear: store byte tokens as single-`bytes`-derived placeholder chars, `<0xNN>` strings, or latin-1-mapped single chars.
   - Recommendation: planner picks one and proves it via a non-ASCII round-trip in the harness; latin-1 single-char mapping of each byte is a common length-faithful choice (1 char per byte) but must not collide with real chars or the marker — decide empirically.

3. **Multi-space / tab / newline handling.**
   - What we know: the *measured* NER `text` is single-space-joined (`' '.join(words)`), so the scored path only sees single spaces. Raw `domain_*_dev.txt` lines (used by `test_tokenizer.py` for efficiency/speed) may contain runs of spaces (verified: Twitter lines have double spaces).
   - What's unclear: whether to emit one `▁` per literal space (length-faithful) or collapse.
   - Recommendation: **one `▁` per literal space** (length-faithful by construction); since the checker's reconstruction is whitespace-insensitive and the NER path is single-space, this is safe. Confirm against raw dev lines in the harness.

4. **Bigram-forcing threshold.**
   - What we know: must guarantee ≥1 legal cross-space bigram deterministically.
   - Recommendation: after normal training, if the checker predicate finds no bigram, force-merge the single highest-count legal cross-space pair (count ties broken lexicographically). Make it always-checked, not threshold-gated.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All local dev | ✓ | 3.11.3 (`py`) | — (in env range 3.11–3.12) |
| `uv` | `init.sh` / env sync / running scripts as `uv run` | ✗ (not on PATH) | — | Use system `py` + a manual venv, OR install `uv`; the **reproducible** path is `uv run python generate_tokenizers.py`, so the planner should ensure `uv` is installed for the real generation/check runs. |
| `.venv` | torch/numpy/regex/tqdm at pinned versions | ✗ (not created) | — | Run `bash init.sh` (installs uv + `uv sync`) as an early Phase-1 task. |
| numpy / regex / torch / tqdm | scaffold + (numpy/regex/tqdm) our code | ✗ until `uv sync` | pinned in `uv.lock` | Provided by `uv sync`; no manual install. |
| Tesla M60 GPU (course VM) | EVAL-01 scored F1 run | ✗ locally | — | CPU runs the NER smoke locally (slow but works); only the final scored F1 number needs the M60. Protect the 10h budget. |

**Missing dependencies with no fallback:** none that block local development of `bpe_tokenizer.py` (Python is present; the tokenizer logic + harness need only numpy/regex/tqdm which `uv sync` provides).
**Missing dependencies with fallback:**
- `uv` / `.venv` not yet set up → run `bash init.sh` early (Phase-1 task 1 alongside zip extraction).
- M60 GPU absent locally → CPU smoke pass locally; reserve the VM strictly for the scored F1 run.

## Validation Architecture

> `workflow.nyquist_validation: true` (verified in `.planning/config.json`) — section required.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | **None present** — no pytest/unittest config in the scaffold. Phase-1 validation is via (a) our own assertion harness `code/align_check.py`, (b) the provided `code/test_tokenizer.py`, (c) `check_submission.py`, (d) `code/train_ner_model.py`. |
| Config file | none — see Wave 0 |
| Quick run command | `py code/align_check.py` (CPU; gates + alignment + pickle round-trip) — or `uv run python code/align_check.py` once env synced |
| Full suite command | `uv run python check_submission.py HW2_214034621.zip` (full gate incl. 1-batch NER smoke) |

### Phase Requirements → Validation Map
Each disqualification gate / correctness invariant is a concrete validation point (VP). "Local" = CPU, no VM. Only the **scored** F1 number needs the M60.

| VP | Behavior | Type | Method / Command | Local CPU? |
|----|----------|------|------------------|-----------|
| VP-1 | ≥1 bigram AND no token with 2+ internal spaces | unit assertion (mirror checker predicate) | `code/align_check.py`: `assert any(t.replace('▁',' ').strip().count(' ')==1 ...)` and `assert not any(... .count(' ')>=2)` | ✅ Local |
| VP-2 | `space_token` non-None and in `token_to_id` | unit assertion | `align_check.py`: `assert tok.space_token is not None and tok.space_token in tok.token_to_id` | ✅ Local |
| VP-3 | Pickle reload from `code/` alone in a FRESH process round-trips encode/decode | subprocess harness | save → spawn new `py` process with cwd at a copied `code/` → `BaseTokenizer.load` → encode/decode equal pre-pickle | ✅ Local |
| VP-4 | Decode length-faithful: cumulative `decode(ids[:i+1])` monotonic + exact char offsets on dev_1/dev_2 samples (incl. non-ASCII & double-space lines); `decode(encode(x)).replace(' ','')==x.replace(' ','')` | local harness mirroring `_get_token_char_spans_fast` | `align_check.py` over N dev lines from both domains | ✅ Local (simulated already in this research) |
| VP-5 | Byte-reproducibility: unedited `generate_tokenizers.py` twice → byte-identical `tokenizer_1.pkl` | double-run + diff | run script twice to two dirs; `cmp tokenizer_1.pkl` | ✅ Local (needs `uv`/env) |
| VP-6 | Zero `[UNK]`: byte fallback → every char maps to in-vocab tokens | unit assertion | encode a non-ASCII / emoji / OOV stress string; assert id 1 (`[UNK]`) never appears | ✅ Local |
| VP-7 | NER **F1 ≥ 0.4** on `dev_1_binary.tagged` via tokenizer 1, end-to-end | full NER pipeline run | `uv run python code/train_ner_model.py --tokenizer_path trained_tokenizers/tokenizer_1.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged` | ⚠️ CPU smoke locally (slow, ~20 epochs BiLSTM); **scored number on M60 VM** |
| VP-8 | Full submission gate: structure + all 3 pkls load + NER smoke pass | checker run on built zip | `uv run python check_submission.py HW2_214034621.zip` | ✅ Local (CPU; smoke is 1 batch) |

### Sampling Rate
- **Per task commit:** `py code/align_check.py` (VP-1,2,3,4,6) — must be green.
- **Per wave merge:** `uv run python check_submission.py HW2_214034621.zip` (VP-8) on a built zip; VP-5 double-run diff.
- **Phase gate:** VP-7 (F1≥0.4) green — CPU smoke must show non-trivial F1 locally before spending the M60; the scored run on the VM is the final confirmation.

### Wave 0 Gaps
- [ ] `code/align_check.py` — new OURS harness covering VP-1,2,3,4,6 (the local gate). Does not exist yet.
- [ ] Fresh-process pickle-load subprocess helper (can be inside `align_check.py`) — VP-3.
- [ ] A tiny seeded subsample path inside `train()` so the harness/check loop runs in seconds (D-04).
- [ ] Env bootstrap: `uv` not installed, `.venv` not created — run `bash init.sh` before any `uv run` / NER / checker step.
- [ ] No framework install needed beyond `uv sync` (no pytest required; plain asserts suffice).

*(Anything that cannot be reliably validated locally: only the **scored** F1 magnitude on the M60 — but a CPU smoke pass of the identical pipeline validates that it runs end-to-end and gives a directional F1, so no gate is invisible until the VM.)*

## Security Domain

> `security_enforcement` is not set in `.planning/config.json` (absent). This is offline, single-user university coursework with **no** untrusted input, network, auth, secrets, or persistence beyond a local pickle. The standard ASVS web categories (V2/V3/V4/V6) do not apply. The only integrity-adjacent concerns are correctness/anti-disqualification, already covered above.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | partial (robustness, not security) | Byte fallback guarantees no input crashes encode/decode (zero UNK); handled by CORE-05. |
| V6 Cryptography | no | — (never hand-roll crypto; none needed) |

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Pickle load of an attacker-controlled file | Tampering (theoretical) | Out of scope — we load only our own self-produced pkl; the grader loads ours. No untrusted pickle. |
| Editing a forbidden graded file (integrity) | Tampering | Keep `base_tokenizer.py`/`train_ner_model.py`/`generate_tokenizers.py`/`check_submission.py` byte-identical to originals; diff before packaging (Pitfall 1 / PKG-01). |

## Sources

### Primary (HIGH confidence)
- **Extracted scaffold source** (this research, from `code_env_example.zip`): `check_submission.py` (bigram predicate :276-292, load mechanism :106-143/295-311, soft-warning train_commands :218-222), `base_tokenizer.py` (specials, save/load pickle :98-123), `train_ner_model.py` (alignment :117-200, fixed HP/seed :18-35), `generate_tokenizers.py` (vocab_size-only constructor, seed=42, default train-file map), `test_tokenizer.py` (whitespace-insensitive reconstruction :90), `train_tokenizer.py`, `README.md`, `pyproject.toml` (cu126/M60), sample `HW2_123456789.zip` (exact required layout). [VERIFIED: local file inspection]
- **Empirical aligner simulation** (this research): README "Steve Jobs" example pushed through the real `_get_token_char_spans_fast` + overlap-alignment logic with a length-faithful decode → correct first-subtoken mapping incl. cross-word bigram. [VERIFIED: executed locally]
- **Data inspection** (this research): line counts (domain_1=1.28M, domain_2=320K), non-ASCII present in dev (Arabic U+0650, smart quotes), `▁` zero-collision in train data, `.tagged` format (TAB-separated, blank-line sentence breaks, binary tag). [VERIFIED: executed locally]
- `.planning/research/ARCHITECTURE.md`, `.planning/research/PITFALLS.md`, `.planning/research/STACK.md` — project-level BPE/alignment/pickle/determinism guidance. [CITED]

### Secondary (MEDIUM confidence)
- SentencePiece (Kudo & Richardson) [arXiv:1808.06226] / [google/sentencepiece] — U+2581 marker, lossless `replace('▁',' ')` detokenization. [CITED]
- tiktoken `_educational.py` — greedy lowest-rank encode (relevant to Phase-2 speed, referenced for Phase-1 correctness baseline). [CITED]

### Tertiary (LOW confidence)
- None new — Phase-1 findings are grounded in verified scaffold inspection, not unverified web claims.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — env fully pinned in provided `pyproject.toml`/`uv.lock`; we add nothing.
- Scaffold contracts (load, alignment, gates, zip layout): HIGH — read from actual source + empirically simulated.
- F1≥0.4 reachability on a simple subsample: MEDIUM — modest floor, but unverified until the run (A1).
- Byte-fallback length-faithful round-trip of non-ASCII: MEDIUM-HIGH — mechanism sound, exact surface representation is an open choice (Open Q2).

**Research date:** 2026-06-25
**Valid until:** ~2026-07-25 (stable; the scaffold is fixed/graded-against-original, so contracts won't drift within the assignment window).
