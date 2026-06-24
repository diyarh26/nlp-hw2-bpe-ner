# Feature Research

**Domain:** From-scratch BPE tokenizer feeding a FIXED binary NER model; 3 variants (Twitter, news, hidden-domain). Competition axes (equal weight): token efficiency (tokens/sentence), encoding speed, NER F1.
**Researched:** 2026-06-24
**Confidence:** MEDIUM-HIGH (gate features = HIGH, anchored in PROJECT.md spec; competition/generalization tactics = MEDIUM, cross-checked against BPE/SentencePiece literature)

> **Axis legend used throughout:** `EFF` = token efficiency (fewer tokens/sentence), `SPD` = encoding speed, `F1` = NER F1, `HID` = hidden-domain generalization, `GATE` = pass/fail correctness gate (disqualification risk). Trade-offs are called out explicitly because the three competition axes pull against each other.

## Feature Landscape

### Table Stakes (Pass-or-Fail — Missing These Means Disqualification or a Zeroed Score)

These are dictated by `check_submission.py`, the NER alignment mechanism, and the F1 gate. They earn no competitive credit but are non-negotiable.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **True char-up BPE** (`train`/`encode`/`decode` from characters) | Spec requires from-scratch BPE subclassing `BaseTokenizer`; not a wrapper | MEDIUM | Base alphabet = all chars (or bytes) seen in training; iteratively merge most-frequent adjacent pair. `GATE` |
| **>=1 cross-word bigram token** (exactly one internal space) | `check_submission.py` disqualifies if zero bigrams. Requires merges allowed to cross the space marker | MEDIUM | The deviation from GPT-2: do NOT pre-tokenize on whitespace into closed words. Space marker must be a *mergeable* symbol. `GATE` `EFF` |
| **Hard cap at bigram (<=1 internal space, never 3+ words)** | Spec forbids 2+ internal spaces; `tok.replace(space_token,' ').strip()` must contain at most one internal space | MEDIUM | Enforce at merge time: reject any candidate merge whose resulting token would contain a second space marker. `GATE` |
| **`space_token` attribute set + present in vocab** | Checked non-`None` and must be in vocab | LOW | Pick a marker that never occurs in raw text (e.g. `▁` U+2581, SentencePiece convention). `GATE` |
| **Length-faithful, reversible decode** | `train_ner_model.py` derives token→word char spans via `decode([id])` and cumulative `decode(token_ids[:i+1])`, comparing lengths to original text | HIGH | `decode(encode(text)) == text` must hold and marker→real-space restoration must be length-preserving. Any off-by-one breaks span alignment and craters F1. `GATE` `F1` |
| **Pickle-safe (no lambdas / unpicklable state)** | Must load via `BaseTokenizer.load` from `code/` alone | LOW | Use plain dicts/ints; module-level functions only; no closures, no `re.compile` of lambdas stored on instance. `GATE` |
| **F1 >= 0.4 on both dev sets** | Hard gate (20% correctness block) | MEDIUM | Driven mostly by alignment correctness + not over-fragmenting entities. `GATE` `F1` |
| **Reproducible via unedited `generate_tokenizers.py` + `train_commands.txt`** | Byte-reproducibility required; seeded | LOW | Deterministic tie-breaking in merge selection (sort pairs deterministically, not dict-order). `GATE` |

### Differentiators (Where the 45% Competition Score and 15% Hidden-Domain Score Are Won)

| Feature | Value Proposition | Complexity | Notes / Axis |
|---------|-------------------|------------|--------------|
| **Cross-word bigram merges tuned for frequency** (merge common collocations: "in the", "@ user", "https :", named-entity pairs) | Directly cuts tokens/sentence; each merged bigram removes ~1 token per occurrence | MEDIUM | `EFF`↑ strongly. **Trade-off:** over-merging entity word-pairs (e.g. "New York" into one token spanning two words) can HELP F1 (entity stays atomic) OR HURT it if the bigram crosses an entity boundary (e.g. "York said") — see Anti-Features. `EFF` `F1`(±) |
| **Byte-level base alphabet with byte fallback (no UNK)** | Guarantees zero OOV/UNK on the hidden domain (unseen emoji, scripts, typos). Literature: byte fallback gave +0.87 F1 on an *unseen* social-media benchmark vs. no fallback | MEDIUM-HIGH | **The single highest-leverage hidden-domain feature.** UNK tokens destroy NER spans (decode length mismatch) and lose entity signal. `HID`↑↑ `F1`(HID)↑. **Trade-off:** byte base inflates token count on rare/non-ASCII text (each unknown char = its UTF-8 bytes) → small `EFF` cost on exotic input, negligible on ASCII. |
| **Tuned vocab size per domain** | Sweet-spot vocab balances efficiency vs. fragmentation; news (formal, lower OOV) tolerates larger vocab; Twitter (noisy) benefits from moderate vocab + fallback | MEDIUM | `EFF`/`F1` both sensitive. Larger vocab → fewer tokens (`EFF`↑) but more, longer tokens that may merge across entity boundaries (`F1` risk) and slower train. Sweep a few sizes on dev. `EFF` `F1` `SPD`(train only) |
| **Fast encode via precomputed merge-rank lookup** (priority-queue / rank-map merge, NOT naive O(n²) full rescan per merge) | Wins the speed axis; encode is the timed operation | MEDIUM-HIGH | Standard fast BPE: assign each merge a rank, repeatedly apply lowest-rank applicable merge on the symbol list (linked-list / index approach). `SPD`↑↑. Pure-Python/numpy only — no `tokenizers` lib allowed. |
| **Per-word (per-segment) encode cache** | Same words recur constantly; cache `segment -> token_ids`. Massive speedup on repetitive corpora (Twitter especially) | LOW-MEDIUM | Dict cache keyed by raw segment string. **Trade-off:** cross-word bigrams mean caching can't be purely per-word — cache per *pre-segment chunk* (e.g. per line or per N-gram window) or cache word-internal pieces then merge across boundaries in a second pass. `SPD`↑↑. Pickle-safe: a plain dict (or rebuild at load). |
| **Domain-specific preprocessing — Twitter (tokenizer 1)**: normalize/segment URLs, `@mentions`, `#hashtags`, emoji as coherent units before/within BPE | Keeps `@user`, `http://...`, `#tag` from fragmenting into many tokens (`EFF`↑) and keeps entity-bearing handles/hashtags atomic (`F1`↑) | MEDIUM | Use `regex` (allowed) for light pre-segmentation. **Must remain length-faithful** — any normalization that changes string length breaks decode alignment, so prefer *segmentation* (where to allow/forbid merges) over *rewriting*. `EFF` `F1` |
| **Domain-specific preprocessing — News (tokenizer 2)**: exploit formal structure (capitalization, common function-word bigrams, punctuation) | Formal text has stable high-frequency collocations → high-value bigram merges; consistent casing aids entity tokens | LOW-MEDIUM | News is "easier": lower OOV, predictable distribution. Larger vocab + aggressive function-word bigrams pay off. `EFF` `F1` |
| **Tokenizer-3 data-mix / balancing for generalization** (mix domains 1+2; balance so neither dominates; dedup near-duplicate lines) | A vocab over-fit to one domain transfers poorly to the unseen third. Balanced + deduped training yields a *general-purpose* merge set | MEDIUM | Domain 1 (~95MB) dwarfs domain 2 (~29MB) → **must downsample/upweight** or domain-1 idioms swamp the merges. Dedup prevents frequent duplicate lines from creating junk high-rank merges. `HID`↑↑. **Trade-off:** generalizing vocab is slightly less efficient on *each* known domain than its specialist. |
| **Merge regularization for transfer** (cap per-merge frequency contribution / prune rare or domain-idiosyncratic merges; favor short, broadly-applicable subwords) | Prevents tokenizer 3 from memorizing domain-1 surface idioms; keeps a robust general subword inventory that fragments gracefully on unseen text | MEDIUM | Concretely: after training, prune merges whose support comes overwhelmingly from one domain, or cap how much any single high-frequency string can dominate merge selection. `HID`↑. **Trade-off:** mild `EFF` loss on known domains. |
| **Deterministic, frequency-ranked bigram reporting** (top-5 / bottom-5 most/least common bigrams) | Required for the 20% report; also a free diagnostic of whether merges are sensible vs. junk | LOW | Report demands "top-5 & bottom-5 most/least common bigrams in training" per tokenizer. Compute from final vocab's bigram tokens ranked by training frequency. `Report`. **Note:** "bottom-5 least common" should still be *valid* bigrams — surface genuinely-rare collocations, useful to discuss over-merging. |

### Anti-Features (Tempting but Harmful — Deliberately Do NOT Build)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Tokens spanning 3+ words / 2+ internal spaces** | "More merging = fewer tokens = better efficiency" | **Instant disqualification** (`check_submission.py`). Also wrecks first-subtoken NER labeling: a 3-word token can straddle an entity boundary, mislabeling 2 words | Hard-cap at exactly one internal space; enforce at merge candidate generation |
| **Aggressive/greedy max-vocab merging** | Bigger vocab → fewer tokens → top efficiency score | Long multi-piece tokens routinely cross entity boundaries → first-subtoken labeling assigns one label to a span that mixes entity+non-entity words → `F1` collapse. Also slows train and over-fits domain | Tune vocab on dev F1 *and* efficiency jointly; stop where F1 starts dropping |
| **UNK token / dropping unknown characters** | Simplifies vocab; "rare chars don't matter" | UNK breaks length-faithful decode → span alignment fails → F1 crash, and on the hidden domain UNK rate spikes (worst place to fail). Disqualification risk if decode isn't length-faithful | Byte-level base + byte fallback → mathematically zero OOV |
| **Length-altering normalization** (lowercasing, unicode NFC/NFKC rewrite, stripping accents, collapsing repeated chars) | "Cleaner text → better merges, smaller vocab" | Any change to string *length or content* breaks `decode(encode(text))==text` and the cumulative-length span derivation → silent F1 destruction. Casing is also entity signal for news NER | Keep transformations strictly length-preserving; if you must normalize, do it as reversible segmentation hints, not rewriting |
| **GPT-2-style whitespace pre-tokenization** (merges never cross spaces) | It's the textbook BPE everyone knows; safe and fast | Produces **zero cross-word bigrams** → fails the bigram gate → disqualification. Also leaves efficiency on the table | Treat space marker as an ordinary mergeable symbol; allow exactly-one-space crossings |
| **Naive O(n²) re-encode** (rescan whole sequence after every merge) | Simplest correct implementation | Loses the speed axis badly on 95MB/29MB corpora and timed encode | Rank-map / linked-list merge application; per-segment cache |
| **Specializing tokenizer 3 to one domain** (e.g. train it mostly on the larger domain-1 file) | "More data = better"; domain 1 is 3x larger | Over-fits to Twitter idioms; transfers poorly to the *unseen* hidden domain (the whole point of tokenizer 3) | Balanced mix + dedup + merge regularization toward general subwords |
| **Lambdas / compiled-regex objects stored on the instance** | Convenient functional style | Breaks pickling → fails `BaseTokenizer.load` → disqualification | Module-level named functions; store regex *patterns as strings*, compile lazily; or rebuild caches in `__setstate__` |
| **External data or extra libraries** | Could improve coverage/speed | Explicitly out of scope → integrity/format violation → grade 0 | Only provided data; only numpy/regex/torch/tqdm |

## Feature Dependencies

```
Length-faithful reversible decode
    └──requires──> space_token marker that is mergeable AND restores to exactly one space
                       └──requires──> NO length-altering normalization
                       └──enables──> NER span alignment (decode-based)
                                          └──enables──> F1 >= 0.4 gate

Cross-word bigram merges
    └──requires──> space marker treated as mergeable symbol (NOT GPT-2 pre-tokenization)
    └──constrained-by──> <=1 internal space cap (bigram gate)
    └──enhances──> token efficiency
    └──conflicts-with──> aggressive entity-boundary-crossing merges (F1 risk)

Byte-level base alphabet + byte fallback
    └──guarantees──> zero UNK / zero OOV
                         └──enables──> hidden-domain robustness (tokenizer 3)
                         └──supports──> length-faithful decode on unseen chars

Fast encode (rank-map)
    └──enhanced-by──> per-segment encode cache
    └──complicated-by──> cross-word bigrams (cache key must span word boundaries)

Tokenizer-3 generalization
    └──requires──> balanced data mix (downsample domain 1) + dedup
    └──requires──> byte fallback (no UNK on unseen domain)
    └──enhanced-by──> merge regularization / pruning domain-idiosyncratic merges
```

### Dependency Notes

- **Decode faithfulness gates everything F1-related:** the NER pipeline reconstructs char spans purely from `decode` lengths. This is the load-bearing invariant — every other feature must preserve it.
- **Cross-word bigrams enhance efficiency but conflict with clean entity alignment:** a bigram that fuses two words is great when both words share a label (or both are non-entity), harmful when it crosses an entity boundary. Tune vocab size to capture high-frequency *function-word* and *intra-entity* bigrams while avoiding boundary-crossing fusions.
- **Byte fallback is prerequisite, not optional, for tokenizer 3:** without it the hidden domain produces UNK/decode-mismatch and the 15% analysis score (and F1) suffers.
- **Per-segment cache conflicts with naive per-word caching** because merges cross word boundaries — cache must key on the actual pre-merge segment unit you encode, or do a two-pass (word-internal cached, then cross-boundary merges).

## MVP Definition

### Launch With (v1 — pass all gates first)

- [ ] Char/byte-up BPE `train`/`encode`/`decode` subclassing `BaseTokenizer` — without this nothing scores
- [ ] `space_token` marker + cross-word bigram merges with <=1-space cap — bigram gate
- [ ] Length-faithful reversible decode — F1 gate + alignment
- [ ] Byte-level base + byte fallback (no UNK) — F1 robustness + hidden-domain foundation
- [ ] Pickle-safe, deterministic, reproducible via `generate_tokenizers.py` — load + reproducibility gates
- [ ] F1 >= 0.4 verified on both dev sets

### Add After Validation (v1.x — compete)

- [ ] Fast rank-map encode + per-segment cache — speed axis (add once correctness is locked)
- [ ] Per-domain vocab-size sweep on dev (efficiency vs. F1 Pareto) — tokenizers 1 & 2
- [ ] Twitter preprocessing (URL/@/#/emoji segmentation, length-preserving) — tokenizer 1
- [ ] News collocation/casing tuning — tokenizer 2

### Future Consideration (v2 — hidden-domain edge + report)

- [ ] Tokenizer-3 balanced data mix + dedup + merge regularization/pruning — hidden-domain score
- [ ] Top-5/bottom-5 bigram extraction + dev F1/efficiency tables per model — report deliverable

## Feature Prioritization Matrix

| Feature | User Value (grade) | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Char/byte-up BPE core | HIGH | MEDIUM | P1 |
| Cross-word bigram + <=1-space cap | HIGH | MEDIUM | P1 |
| Length-faithful reversible decode | HIGH | HIGH | P1 |
| Byte fallback (no UNK) | HIGH | MEDIUM | P1 |
| Pickle-safe + reproducible | HIGH | LOW | P1 |
| Fast rank-map encode | HIGH | MEDIUM-HIGH | P2 |
| Per-segment encode cache | MEDIUM | LOW | P2 |
| Per-domain vocab tuning | HIGH | MEDIUM | P2 |
| Twitter preprocessing | MEDIUM | MEDIUM | P2 |
| News collocation tuning | MEDIUM | LOW | P2 |
| Tokenizer-3 mix/dedup/regularization | HIGH (15% block + F1) | MEDIUM | P2 |
| Top/bottom-5 bigram reporting | MEDIUM (report) | LOW | P3 |

## Competitor Feature Analysis (reference tokenizers, for design contrast)

| Feature | GPT-2 BPE | SentencePiece (BPE/Unigram) | Our Approach |
|---------|-----------|------------------------------|--------------|
| Whitespace handling | Pre-tokenize on spaces; merges never cross | `▁` (U+2581) meta-symbol, whitespace is a normal mergeable symbol, lossless reversible decode | `▁`-style mergeable marker, allow cross-word merges but cap at one space (gate-driven) |
| Base alphabet / OOV | Byte-level → no UNK | Char + optional byte fallback → no UNK | Byte-level + byte fallback → no UNK (critical for hidden domain) |
| Decode | Reversible | Guarantees `decode(encode(x))==x` | Length-faithful reversible (required for NER span alignment) |
| Cross-word tokens | Never | Possible (whitespace is mergeable) | **Required** (>=1 bigram) but capped at bigram |
| Cross-domain generalization | N/A (single corpus) | Language-agnostic design | Explicit balanced mix + dedup + merge regularization (tokenizer 3) |

## Key Hidden-Domain (Tokenizer 3) Generalization Strategies — Concrete

1. **Byte-level base + byte fallback** → zero UNK on unseen scripts/emoji/typos; verified in literature to add ~+0.87 F1 on an unseen social-media benchmark.
2. **Balanced data mix**: domain 1 (~95MB) is ~3x domain 2 (~29MB); downsample domain 1 (or upweight domain 2) so merges reflect both registers, not just Twitter idioms.
3. **Deduplication** of near-identical lines before counting pair frequencies → stops frequent duplicate sentences from minting junk high-rank merges that don't generalize.
4. **Merge regularization / pruning**: drop or down-rank merges whose frequency support is overwhelmingly single-domain; favor short, broadly-applicable subwords so unseen text fragments *gracefully* rather than into bytes.
5. **Moderate vocab size** (vs. tokenizer 2's larger formal vocab): smaller, more general subword inventory transfers better to an unknown distribution.
6. **Length-faithful decode preserved** so that even byte-fallback tokens on exotic hidden-domain characters still align for NER.

## Sources

- [BPE for NER / first-subtoken labeling & boundary alignment (arXiv 1809.08386; HF token classification)](https://huggingface.co/docs/transformers/main/tasks/token_classification) — MEDIUM
- [A Byte-sized Approach to Named Entity Recognition (arXiv 1809.08386)](https://arxiv.org/abs/1809.08386) — MEDIUM
- [SentencePiece: lossless reversible tokenization, `▁` meta-symbol (arXiv 1808.06226)](https://arxiv.org/pdf/1808.06226) — MEDIUM-HIGH
- [google/sentencepiece (space marker & decode guarantees)](https://github.com/google/sentencepiece) — MEDIUM-HIGH
- [SMILE: byte fallback +0.87 F1 on unseen social-media domain (arXiv 2307.00135)](https://arxiv.org/pdf/2307.00135) — MEDIUM
- [Data Mixture Inference: BPE merges reveal training-data mix (arXiv 2407.16607)](https://arxiv.org/html/2407.16607v2) — MEDIUM
- [Hugging Face tokenizer_summary (byte-level BPE, no OOV)](https://huggingface.co/docs/transformers/tokenizer_summary) — MEDIUM
- `.planning/PROJECT.md` (gate spec, NER alignment mechanism, bigram check detail) — HIGH (project ground truth)

---
*Feature research for: BPE-for-cross-domain-NER tokenizer (NLP HW2)*
*Researched: 2026-06-24*
