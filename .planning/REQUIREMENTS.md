# Requirements: BPE Tokenizer for Cross-Domain NER (NLP HW2)

**Defined:** 2026-06-24
**Core Value:** Three correct, reproducible BPE tokenizers that pass every hard gate and score well on the competition's three axes (efficiency, speed, F1), including the hidden domain.

## v1 Requirements

Committed scope for the submission. Each maps to a roadmap phase.

### Core BPE Engine (CORE)

- [x] **CORE-01**: Tokenizer trains true BPE from the character level upward, merging the most frequent adjacent pair iteratively until `vocab_size` is reached.
- [x] **CORE-02**: A `space_token` marker (e.g. `▁`) represents whitespace and is a first-class, mergeable symbol added to the vocabulary (and set as the `space_token` attribute, never `None`).
- [x] **CORE-03**: Merges are allowed to cross the space marker, producing at least one **bigram** token (a token whose surface spans exactly two whitespace-separated words).
- [x] **CORE-04**: Merges are hard-capped at one internal space — the tokenizer never produces a token spanning 3+ words (no trigram+).
- [x] **CORE-05**: A byte/char-level base alphabet with fallback guarantees zero `[UNK]` on any input, including unseen characters in the hidden domain.

### Decode / NER Alignment Contract (ALIGN)

- [x] **ALIGN-01**: `decode` restores real spaces from the marker and is length-faithful, so the NER pipeline's `decode([id])` + cumulative `decode(ids[:i+1])` char-span alignment is correct.
- [x] **ALIGN-02**: `decode(encode(text))` reconstructs the text (matching the checker's whitespace-insensitive reconstruction test) on dev samples of both domains.
- [x] **ALIGN-03**: A local alignment harness verifies token→word mapping and first-subtoken labeling behave correctly before any VM run.

### Serialization & Reproducibility (SERDE)

- [x] **SERDE-01**: The trained tokenizer pickles and unpickles from `code/` alone — only plain dict/list/tuple/str/int state, no lambdas/closures/`defaultdict(lambda)`/local classes.
- [x] **SERDE-02**: The regex is stored as a pattern string (not a compiled `Pattern`); the runtime encode cache is dropped on pickle via `__getstate__`/`__setstate__` and recompiled on load.
- [x] **SERDE-03**: All three tokenizers are byte-reproducible via the unedited `generate_tokenizers.py` with a fixed seed.
- [ ] **SERDE-04**: `train_commands.txt` at the zip root lists the exact commands (vocab sizes, `--train_files_3`, etc.) used to produce `tokenizer_{1,2,3}.pkl`.

### Encode Speed & Training Efficiency (SPEED)

- [x] **SPEED-01**: `encode` uses a merge-rank greedy strategy (rank dict `{(a,b):rank}`), not a naive O(vocab×n) whole-string rescan.
- [x] **SPEED-02**: A per-chunk (per-word) encode cache memoizes repeated segments to exploit Zipfian text.
- [x] **SPEED-03**: Training uses weighted word-frequency chunks + incremental pair-count deltas with an inverted `pair→chunks` index (no full `Counter` recount per merge), so the 95MB/29MB corpora train in reasonable time.

### Per-Domain Tuning & Competition (TUNE)

- [ ] **TUNE-01**: Tokenizer 1 is trained/tuned on the Twitter domain (`domain_1_train.txt`).
- [ ] **TUNE-02**: Tokenizer 2 is trained/tuned on the news domain (`domain_2_train.txt`).
- [ ] **TUNE-03**: Vocab size is treated as a swept hyperparameter; a dev-set evaluation harness scores all three axes (tokens/char efficiency, tokens/sec speed, NER F1) to pick per-domain configs.
- [ ] **TUNE-04**: Final tokenizer configs balance the efficiency/speed/F1 trade-off (aggressive enough for efficiency without wrecking entity-boundary alignment).

### Tokenizer 3 / Hidden-Domain Generalization (GEN)

- [ ] **GEN-01**: Tokenizer 3 trains on a deliberate mix of domain 1+2 data designed to generalize (balanced — downsample the larger domain — and deduplicated near-duplicate lines).
- [ ] **GEN-02**: Tokenizer 3 uses a moderate vocab + byte fallback so an unseen domain incurs no UNK and degrades gracefully.
- [ ] **GEN-03**: A reasoned, evidence-backed evaluation of tokenizer 3's expected hidden-domain behavior is produced for the report.

### Validation & Evaluation (EVAL)

- [x] **EVAL-01**: NER F1 ≥ 0.4 on `dev_1_binary.tagged` (domain 1) using tokenizer 1.
- [ ] **EVAL-02**: NER F1 ≥ 0.4 on `dev_2_binary.tagged` (domain 2) using tokenizer 2.
- [ ] **EVAL-03**: Tokenizer efficiency (tokens/char or tokens/sentence) and tokenization speed are measured and reported on the dev files for each tokenizer.
- [ ] **EVAL-04**: All GPU NER runs are executed on the course VM within the 10h budget, after local gates are green.

### Submission & Report (PKG)

- [x] **PKG-01**: The submission zip (`HW2_214034621.zip`) matches the required structure exactly and passes `check_submission.py` (code/, trained_tokenizers/tokenizer_{1,2,3}.pkl, report, train_commands.txt).
- [ ] **PKG-02**: `report_214034621.pdf` meets the format rules (≤1 A4 page, Arial 10, 1.15 spacing, 2.54cm margins, single column, captioned centered images, in-image text ≥9pt).
- [ ] **PKG-03**: Report contains all mandatory content: author (Diyar Husayyan, 214034621); per-tokenizer explanation (deviation from basic BPE + hidden-domain handling); top-5 & bottom-5 most/least common training bigrams; dev F1 + efficiency per model; hidden-domain evaluation; model-2 performance + parameter/characteristic/improvement discussion.

## v2 Requirements

Stretch optimizations — pursued only if v1 is solid and time/budget allow.

### Advanced Optimization (OPT)

- **OPT-01**: Heap/linked-list encode optimization (O(n log n)) if profiling shows the greedy loop is the speed bottleneck.
- **OPT-02**: Merge regularization / vocab pruning of single-domain idioms to further aid tokenizer-3 transfer.
- **OPT-03**: Domain-specific preprocessing for Twitter artifacts (URLs/@mentions/hashtags/emoji) if it measurably helps efficiency or F1 without breaking decode faithfulness.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Editing `base_tokenizer.py`, `train_ner_model.py`, `generate_tokenizers.py`, `check_submission.py` | Graded against originals; only file-path args in train_ner_model may change |
| External training data or new Python libraries | Spec forbids; only provided data + uv env (numpy/regex/torch/tqdm) |
| Changing fixed NER hyperparameters (seed, batch, LR, epochs) | Model must be uniform across all students |
| Reading/parsing the NER `.tagged` files ourselves | Handled by the provided pipeline |
| GPT-2-style whitespace pre-tokenization | Would produce zero bigrams → disqualification |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-01 | Phase 1 | Complete in 01-02 |
| CORE-02 | Phase 1 | Complete in 01-02 |
| CORE-03 | Phase 1 | Complete in 01-02 |
| CORE-04 | Phase 1 | Complete in 01-02 |
| CORE-05 | Phase 1 | Complete in 01-02 |
| ALIGN-01 | Phase 1 | Complete in 01-02 |
| ALIGN-02 | Phase 1 | Complete in 01-02 |
| ALIGN-03 | Phase 1 | Complete in 01-02 |
| SERDE-01 | Phase 1 | Complete in 01-03 |
| SERDE-02 | Phase 1 | Complete in 01-03 |
| SERDE-03 | Phase 1 | Complete in 01-03 |
| EVAL-01 | Phase 1 | Complete in 01-04 |
| PKG-01 | Phase 1 | Complete in 01-04 |
| SPEED-01 | Phase 2 | Complete in 02-02 |
| SPEED-02 | Phase 2 | Complete in 02-02 |
| SPEED-03 | Phase 2 | Complete in 02-03 |
| TUNE-01 | Phase 3 | Pending |
| TUNE-02 | Phase 3 | Pending |
| TUNE-03 | Phase 3 | Pending |
| TUNE-04 | Phase 3 | Pending |
| GEN-01 | Phase 4 | Pending |
| GEN-02 | Phase 4 | Pending |
| GEN-03 | Phase 4 | Pending |
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |
| EVAL-04 | Phase 5 | Pending |
| SERDE-04 | Phase 5 | Pending |
| PKG-02 | Phase 6 | Pending |
| PKG-03 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 29 total (CORE 5, ALIGN 3, SERDE 4, SPEED 3, TUNE 4, GEN 3, EVAL 4, PKG 3)
- Mapped to phases: 29 (100%) ✓
- Unmapped: 0

**Per-phase counts:** Phase 1 = 13, Phase 2 = 3, Phase 3 = 4, Phase 4 = 3, Phase 5 = 4, Phase 6 = 2.

---
*Requirements defined: 2026-06-24*
*Last updated: 2026-06-24 after roadmap traceability mapping (6 phases)*
