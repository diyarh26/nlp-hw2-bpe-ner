# Architecture Research

**Domain:** From-scratch BPE tokenizer plugged into a FIXED course-provided NER pipeline (NLP HW2)
**Researched:** 2026-06-24
**Confidence:** HIGH (scaffold contracts are fully specified in PROJECT.md; BPE/SentencePiece mechanics corroborated across the official SentencePiece paper/repo and multiple BPE implementation sources)

> Scope note: This is a self-contained, single-module engineering problem, not a multi-service system. "Architecture" here means: the internal structure of `BPETokenizer`, the contracts it must honor with the unedited scaffold, the data flow from training corpus → merges → encode → NER alignment, and the build order / local-vs-VM split under a 10h GPU budget. There is no "scaling to millions of users" dimension; the scaling axis is corpus size (95MB + 29MB) and the competition metrics (speed / efficiency / F1).

---

## Standard Architecture

### System Overview

The boxed region is the ONLY code we own (`code/bpe_tokenizer.py` + helper scripts we add). Everything outside the box is the fixed, graded-against-original scaffold we must not edit.

```
┌───────────────────────────────────────────────────────────────────────┐
│  FIXED SCAFFOLD (do not edit)                                          │
│  generate_tokenizers.py ──imports──┐        train_ner_model.py         │
│  (seed=42, --vocab_size,           │        (BaseTokenizer.load,       │
│   --train_files_{1,2,3})           │         NERDataset alignment,     │
│                                    │         BiLSTM, fixed seed/HP)    │
│  check_submission.py               │        test_tokenizer.py          │
│  (zip layout, space_token!=None,   │        (speed, tokens/char,       │
│   >=1 bigram, smoke NER)           │         reconstruction rate)      │
│  base_tokenizer.py (ABSTRACT) ─────┤                                   │
│   token_to_id/id_to_token,         │                                   │
│   specials {PAD0,UNK1,BOS2,EOS3},  │                                   │
│   encode_batch/decode_batch,       │                                   │
│   get_vocab_size, save/load(pickle)│                                   │
│  ┌─────────────────────────────────▼───────────────────────────────┐  │
│  │  OUR CODE: code/bpe_tokenizer.py                                 │  │
│  │  class BPETokenizer(BaseTokenizer)                               │  │
│  │  ┌───────────┐  ┌────────────┐  ┌──────────┐  ┌──────────────┐  │  │
│  │  │  train()  │→ │ merge table│→ │ encode() │→ │  decode()    │  │  │
│  │  │ (corpus → │  │ + vocab /  │  │ (text →  │  │ (ids → text, │  │  │
│  │  │  merges)  │  │ id maps +  │  │  ids,    │  │ length-      │  │  │
│  │  │           │  │ space_token│  │ heap/LL) │  │ faithful)    │  │  │
│  │  └───────────┘  └────────────┘  └──────────┘  └──────────────┘  │  │
│  │  + helper scripts we add to code/ (local unit tests, tuning)     │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
        │ pickle (.save)                          ▲ decode([id]) per token
        ▼                                         │ + cumulative decode(ids[:i+1])
  trained_tokenizers/tokenizer_{1,2,3}.pkl ───────┘  char-span alignment
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `train(corpus)` | Learn merge table from chars upward; build vocab + id maps; set `space_token`; produce ≥1 cross-word **bigram** token; cap merges at one internal space | Pair-frequency counting over a space-marked symbol sequence; greedy most-frequent-pair merges until `vocab_size` reached |
| Merge table | Ordered list of learned merges `(a,b) -> ab` with rank; drives encode | `list[tuple[str,str]]` plus `dict[(str,str)->rank]` (rank = merge order) |
| `vocab` / id maps | `token_to_id` / `id_to_token` populated on top of the 4 fixed specials (PAD0/UNK1/BOS2/EOS3) | Plain dicts; ids assigned deterministically by merge order |
| `space_token` | Non-`None` marker symbol (SentencePiece-style `▁`, U+2581) that is itself a mergeable vocab entry; absence/None = disqualification | A single Unicode char added to vocab; spaces in text mapped to it before merging |
| `encode(text)` | text → token-id list; fast (avoid naive O(n²) rescan) | Pre-tokenize to char+marker symbols, then apply merges via linked-list + rank heap |
| `decode(ids)` | ids → text; **length-faithful**, restores real spaces | `''.join(id_to_token[i] for i in ids).replace(space_token, ' ')` (+ strip specials) |
| Serialized state | Everything needed to reload via `BaseTokenizer.load` (pickle) — no lambdas/closures | Pure dicts/lists/strings as instance attributes |

---

## Recommended Project Structure

We can only ship `code/bpe_tokenizer.py` plus helper scripts; the forbidden files are excluded from the zip. Local working tree:

```
NLP-HW2/
├── code/
│   ├── base_tokenizer.py          # FIXED — present locally, NOT in zip
│   ├── bpe_tokenizer.py           # OURS — the entire tokenizer
│   ├── train_ner_model.py         # FIXED — local run only, NOT in zip
│   ├── test_tokenizer.py          # FIXED helper (speed/efficiency/recon)
│   ├── train_tokenizer.py         # FIXED helper
│   └── dev_*.py / tune_*.py       # OURS — local unit tests, tuning, EDA
│                                  #        (these helpers MAY ship in code/)
├── generate_tokenizers.py         # FIXED (root) — produces the 3 pkls
├── check_submission.py            # FIXED (root) — local gate
├── data/                          # provided corpora + dev + .tagged
├── trained_tokenizers/            # output: tokenizer_{1,2,3}.pkl
├── train_commands.txt             # exact reproducible commands (zip root)
└── report_214034621.pdf           # ≤1 page report
```

### Structure Rationale

- **`bpe_tokenizer.py` is the only logic file.** Keep all tokenizer state as picklable instance attributes; do not stash helpers in module-level closures or lambdas (breaks `BaseTokenizer.load`).
- **Helper scripts live in `code/`** so they can ship (the zip's `code/` may contain `bpe_tokenizer.py` + edited helpers). Keep them import-light and dependency-free (numpy/regex/torch/tqdm only).
- **`train_commands.txt` is the reproducibility contract** — it records the exact `generate_tokenizers.py` invocation (vocab sizes per tokenizer, `--train_files_3` mix, seed). Graders regenerate the pkls from it.

---

## Architectural Patterns

### Pattern 1: SentencePiece-style space marker for length-faithful decode

**What:** Before any merging, replace runs of whitespace with a literal marker char `space_token = "▁"` (U+2581) that is added to the vocab as a real, mergeable symbol. Decode is the inverse: `''.join(tokens).replace("▁", " ")`.
**When to use:** Mandatory here — the NER pipeline derives token→word char spans by comparing `decode([id])` and cumulative `decode(ids[:i+1])` lengths against the original text. Decode MUST reproduce spaces at the right positions or alignment silently corrupts labels.
**Trade-offs:** A single marker char is simple and lossless for single spaces; runs of multiple spaces / tabs / newlines need a deliberate normalization decision (see Anti-Patterns). The marker must satisfy `check_submission.py`'s bigram test: a token is a bigram iff `tok.replace(space_token,' ').strip()` still has an internal space — so the marker must map cleanly back to one space.

**Example:**
```python
def decode(self, ids):
    pieces = [self.id_to_token[i] for i in ids
              if i not in (self.special_tokens["[PAD]"], self.special_tokens["[BOS]"],
                           self.special_tokens["[EOS]"])]
    return "".join(pieces).replace(self.space_token, " ")
```

### Pattern 2: Cross-word merges, hard-capped at one internal space (the "bigram" rule)

**What:** Unlike GPT-2 (which pre-tokenizes on whitespace and never merges across spaces), we ALLOW a merge to cross exactly one `space_token`, producing a single token spanning two words (a bigram). We FORBID any merge whose result would contain two or more `space_token`s.
**When to use:** Required — `check_submission.py` demands ≥1 bigram token (else disqualification) and forbids 3+ word tokens. Cross-word merges also reduce token count (the efficiency metric).
**Trade-offs:** Bigram tokens raise efficiency and can help speed, but over-aggressive cross-word merging can hurt NER alignment (a bigram token that straddles a word boundary still aligns fine via char spans, but rare/over-fit bigrams waste vocab). Cap merges by rejecting any candidate pair whose merged string contains ≥2 markers.

**Example:**
```python
def _mergeable(self, a, b):
    merged = a + b
    return merged.count(self.space_token) <= 1   # cap at one internal space
```

### Pattern 3: Linked-list + rank-heap encode (O(n log n), not O(n²))

**What:** Represent the symbol sequence as a doubly-linked list; maintain a priority queue keyed by merge rank (merge order learned in `train`). Repeatedly pop the lowest-rank applicable adjacent pair, merge in place, and push the two newly-formed neighbor pairs. No full rescans, no array shifting.
**When to use:** To win the speed axis on 95MB/29MB corpora and large dev sets. The naive "scan whole sequence for best pair, repeat" loop is O(n²) and will lose the speed metric.
**Trade-offs:** More code and care (stale heap entries must be validated against current links). A simpler correct fallback: apply merges in learned-rank order with a single left-to-right pass per rank using `regex`/string ops — slower but easy to verify; start here, optimize once correct.

**Example:**
```python
# heap entries: (rank, position_id); validated against live linked list on pop
# merge applies only if both nodes still adjacent and unmerged
```

---

## Data Flow

### Training → Serialized tokenizer

```
domain corpus (one sentence/line)
    ↓  normalize + map spaces → space_token ("▁")
char/marker symbol sequences
    ↓  count adjacent-pair frequencies
greedy pick most-frequent mergeable pair (count markers ≤ 1)
    ↓  record merge (a,b)->ab with rank; add ab to vocab
repeat until len(vocab) == vocab_size
    ↓
merge table + token_to_id/id_to_token (on top of PAD0/UNK1/BOS2/EOS3) + space_token
    ↓  generate_tokenizers.py calls .save() (pickle)
trained_tokenizers/tokenizer_{1,2,3}.pkl
```

### Encode → NER alignment (the load-bearing flow)

```
original sentence text
    ↓  encode(): map spaces→▁, split to symbols, apply merges (heap/LL)
token-id list
    ↓  train_ner_model.py NERDataset:
    │    for each token i: decode([id_i])  → its surface string
    │    cumulative decode(ids[:i+1])      → running char offset
    │    compare lengths/char spans vs original text
    │    → assign word label to FIRST subtoken of each word, rest = -100
aligned (token, label) pairs → BiLSTM → precision/recall/F1
```

The alignment is entirely length-driven. If `decode` is not length-faithful (drops/duplicates a space, normalizes case, collapses whitespace differently than encode), cumulative char offsets drift and labels land on the wrong tokens — F1 silently collapses even though encode "works".

### Reproducibility flow

```
train_commands.txt (vocab sizes, --train_files_3 mix, --seed 42)
    ↓ grader runs unedited generate_tokenizers.py
identical tokenizer_{1,2,3}.pkl (byte-reproducible)
```

---

## Build Order & Local/VM Split

This replaces the generic "scaling" section — the real constraints are dependency order and the **10h GPU budget** on the course VM (Tesla M60). **Everything except NER F1 runs locally on Windows CPU.**

| Stage | Where | Depends on | Output / Gate |
|-------|-------|-----------|---------------|
| 0. Env + data setup | Local (uv) | — | `uv` env (numpy/regex/torch/tqdm); unzip data; confirm `generate_tokenizers.py`/`check_submission.py` import our stub |
| 1. Core BPE `train` | Local CPU | 0 | Merge table + vocab build on a SMALL corpus slice; set `space_token`; produce ≥1 bigram |
| 2. encode/decode correctness | Local CPU | 1 | `decode(encode(x))` reconstruction (per `test_tokenizer.py`'s `.replace(' ','')` rule); **length-faithfulness assert** (see verification below) |
| 3. Pickle round-trip | Local CPU | 1–2 | `.save()` then `BaseTokenizer.load()` → identical encode/decode; no lambdas/closures |
| 4. Alignment smoke test | Local CPU | 2–3 | Run `train_ner_model.py` for a few steps on dev to confirm spans align (CPU is fine for a smoke pass) |
| 5. Speed optimization | Local CPU | 2 | Linked-list + heap encode; benchmark tokens/sec vs naive baseline |
| 6. 3 variants + vocab tuning | Local CPU | 1–5 | Tune vocab_size per domain; choose `--train_files_3` mix; regenerate via `generate_tokenizers.py`; run `check_submission.py` |
| 7. **NER F1 evaluation** | **VM GPU** | 1–6 GREEN locally | Full `train_ner_model.py` on dev_1/dev_2 → F1 ≥ 0.4 gate + competition numbers |
| 8. Report + packaging | Local | 7 | ≤1-page report (top/bottom-5 bigrams, F1+efficiency, hidden-domain analysis); build `HW2_214034621.zip`; final `check_submission.py` |

**10h-budget discipline (HIGH importance):**
- Only stage 7 needs the GPU. Get stages 1–6 fully green locally first — a broken `decode` or pickle failure discovered on the VM burns irreplaceable budget.
- Run `check_submission.py` and a CPU NER smoke pass (stage 4) locally BEFORE any VM run, so the VM only ever sees code that already trains end-to-end.
- Batch VM work: train all three tokenizers locally, copy the three pkls + commands to the VM, run the full NER eval for tokenizer_1 and tokenizer_2 (and hidden-domain checks) in one session. Watch `time_left`.
- Treat each VM NER run as expensive: log F1 + efficiency to a file so you never re-run just to re-read a number.

---

## Anti-Patterns

### Anti-Pattern 1: GPT-2-style pre-tokenization (split on whitespace, never merge across it)

**What people do:** Copy the standard GPT-2/BPE recipe that pre-splits text on whitespace so merges stay within a word.
**Why it's wrong:** It can NEVER produce a bigram token → `check_submission.py` fails → disqualification. It also leaves efficiency on the table.
**Do this instead:** Map spaces to `space_token` and allow merges to cross exactly one marker, hard-capped at one internal space.

### Anti-Pattern 2: Non-length-faithful decode (whitespace collapse, normalization, special-token leakage)

**What people do:** Strip/normalize whitespace, lowercase, or emit `[BOS]`/`[EOS]`/`[PAD]` surface text inside `decode`, or `.strip()` the output.
**Why it's wrong:** `train_ner_model.py` aligns by comparing cumulative `decode(ids[:i+1])` lengths to the original text. Any char-count drift misaligns labels and silently tanks F1 — the failure looks like a bad model, not a bad decode.
**Do this instead:** `decode` = exact inverse of encode at the character level: join surface pieces, replace `space_token`→one space, exclude special tokens from surface output, and ensure multi-space/tab/newline handling in encode is mirrored in decode. Verify with the assert below.

### Anti-Pattern 3: Unpicklable tokenizer state

**What people do:** Store compiled `regex` objects only if they don't survive pickle, lambdas, closures, or generators as instance attributes.
**Why it's wrong:** `BaseTokenizer.load` is a pickle classmethod; `generate_tokenizers.py` saves and the NER pipeline loads. Unpicklable state = load crash = zero usable tokenizers.
**Do this instead:** Keep state as plain dicts/lists/strings/ints. Rebuild any compiled regex in `__init__`/lazily, or recompile on first use after load.

### Anti-Pattern 4: Tuning/optimizing on the VM

**What people do:** Iterate on vocab size, merge logic, or speed on the GPU VM "because that's where F1 is."
**Why it's wrong:** Burns the 10h budget on work that is 100% CPU-bound (tokenizer training/encode are CPU; only NER training needs GPU).
**Do this instead:** All tokenizer dev, unit tests, `check_submission.py`, and a CPU NER smoke pass run locally; the VM only runs the final full NER F1 evaluation.

---

## Verification: decode length-faithfulness (alignment safety)

Because NER alignment is purely length/char-span driven, add a local invariant test BEFORE any VM run:

```python
def assert_length_faithful(tok, text):
    ids = tok.encode(text)
    # 1. full round-trip reconstructs spacing (test_tokenizer.py uses replace(' ',''))
    assert tok.decode(ids).replace(" ", "") == text.replace(" ", "")
    # 2. cumulative prefix decode is monotonic & matches a prefix of the text's char stream
    prev_len = 0
    for i in range(len(ids)):
        prefix = tok.decode(ids[:i+1])
        assert len(prefix) >= prev_len          # never shrinks
        assert text.startswith(prefix) or prefix.replace(" ", "") == text[:len(prefix)].replace(" ", "")
        prev_len = len(prefix)
    # 3. per-token surface is non-empty for non-special ids (spans must advance)
    for i in ids:
        if i not in (0,2,3):  # PAD/BOS/EOS
            assert len(tok.decode([i])) >= 1
```

Run this over a sample of real dev sentences (including Twitter `@mentions`/URLs/emoji and news punctuation) from both domains. If the cumulative-prefix check ever fails, the NER aligner will mislabel — fix `decode`/`encode` symmetry before spending VM time.

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| (none) | Self-contained; no network, no external APIs | Only the local `uv` env + provided data |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `BPETokenizer` ↔ `BaseTokenizer` | Subclass; implement `train`/`encode`/`decode`, set `space_token`, populate `token_to_id`/`id_to_token` on top of fixed specials | Inherit `save`/`load`/`encode_batch`/`decode_batch`/`get_vocab_size` unchanged |
| `BPETokenizer` ↔ `generate_tokenizers.py` | `generate_tokenizers.py` imports our class, calls `train()` + `save()` with seed=42 and per-domain vocab sizes | Do not edit it; control behavior only via its CLI args recorded in `train_commands.txt` |
| `tokenizer_*.pkl` ↔ `train_ner_model.py` | `BaseTokenizer.load` → `encode`/`decode` for NER alignment | The decode-length contract lives entirely here |
| `BPETokenizer` ↔ `check_submission.py` | Validates `space_token!=None`, ≥1 bigram, zip layout, smoke NER | Local pre-flight gate before every submission build |

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Scaffold contracts (specials, save/load pickle, alignment-by-decode-length, bigram check semantics) | HIGH | Directly specified in the fixed-scaffold description and PROJECT.md |
| SentencePiece `▁`/U+2581 marker → lossless, length-faithful decode | HIGH | Corroborated by the official SentencePiece paper (arXiv:1808.06226) and repo; matches the project's stated decision |
| Cross-word merge + one-internal-space bigram cap | HIGH | Derived from `check_submission.py`'s documented bigram test (`tok.replace(space_token,' ').strip()` has internal space) |
| Linked-list + rank-heap O(n log n) encode | MEDIUM-HIGH | Standard, well-documented BPE optimization (Rust NLP / formal BPE papers); single-provider websearch but multi-source corroborated. Recommend starting with a correct O(n·merges) version, then optimizing |
| Build order + local/VM 10h split | HIGH | Follows directly from the stated compute constraints (tokenizer = CPU, NER F1 = GPU, 10h budget) |

## Sources

- SentencePiece (Kudo & Richardson) — [arXiv:1808.06226](https://arxiv.org/abs/1808.06226), [google/sentencepiece](https://github.com/google/sentencepiece): U+2581 whitespace meta-symbol, lossless `''.join(pieces).replace('▁',' ')` detokenization.
- [Byte Pair Encoding and Data Structures — Rust NLP tales](https://guillaume-be.github.io/2021-09-16/byte_pair_encoding): linked-list + priority-queue encode avoiding O(n²) rescans.
- [A Formal Perspective on Byte-Pair Encoding (arXiv:2306.16837)](https://arxiv.org/abs/2306.16837): complexity of BPE merge application.
- `.planning/PROJECT.md` and the fixed-scaffold spec (this assignment): authoritative source for all contract details.

---
*Architecture research for: from-scratch BPE tokenizer in a fixed NER scaffold (NLP HW2)*
*Researched: 2026-06-24*
