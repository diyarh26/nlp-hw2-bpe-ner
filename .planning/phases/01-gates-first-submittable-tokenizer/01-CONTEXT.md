# Phase 1: Gates-First Submittable Tokenizer - Context

**Gathered:** 2026-06-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a complete-but-minimal `code/bpe_tokenizer.py` (`BPETokenizer(BaseTokenizer)`) that:
- implements true char-up BPE `train`/`encode`/`decode`,
- passes every hard disqualification gate in `check_submission.py` (≥1 cross-word bigram, no 3+-word token, `space_token` non-None and in vocab, pickle-reloadable from `code/` alone, correct zip structure),
- decodes **length-faithfully** so the fixed NER aligner maps tokens→words correctly,
- is byte-reproducible via the **unedited** `generate_tokenizers.py` + `train_commands.txt`,
- and clears **F1 ≥ 0.4 on `dev_1_binary.tagged`** end-to-end.

The output is a genuinely submittable `HW2_214034621.zip`.

**Explicitly NOT in this phase:** encode-speed / training-efficiency optimization (Phase 2), per-domain vocab tuning (Phase 3), tokenizer-3 hidden-domain generalization (Phase 4), the VM F1 run for domain 2 (Phase 5), the report (Phase 6). Covers requirements CORE-01..05, ALIGN-01..03, SERDE-01..03, EVAL-01, PKG-01.
</domain>

<decisions>
## Implementation Decisions

### Base Alphabet & UNK Handling (CORE-01, CORE-05)
- **D-01:** Base alphabet is **char-level** — the set of characters observed in the (subsampled) training data become the atomic base tokens, plus all 256 single-byte values to back the fallback.
- **D-02:** **Zero UNK via UTF-8 byte fallback** — any character not in the char vocab (e.g. an unseen glyph in the hidden domain) is decomposed into its UTF-8 byte tokens (which are always in the vocab). `[UNK]` is never emitted by `encode`. Rationale: keeps the load-bearing length-faithful decode **exact** on domains 1 & 2 (where F1 is measured and unseen chars are essentially absent), while still guaranteeing graceful degradation on the truly-hidden domain. Chosen over full byte-level BPE specifically to keep the F1-critical aligner simple/exact on the measured domains.

### Phase-1 Build Scope / Training Volume (CORE-01)
- **D-03:** **Subsample `domain_1_train.txt` for Phase 1.** Train the correct-but-simple BPE on a capped subset (e.g. first N sentences / N MB) so a naive trainer runs in seconds–minutes and yields the gates + F1≥0.4 signal fast. The full-corpus, efficient/incremental trainer is **Phase 2** — the final reproducible tokenizers are produced after that.
- **D-04:** The subsample must be **deterministic/seeded** so the Phase-1 tokenizer is still byte-reproducible (SERDE-03). Note for planner: `generate_tokenizers.py` only passes `vocab_size` to the constructor and reads the full file, so any Phase-1 cap must live **inside** `BPETokenizer.train()` (e.g. a seeded line cap / constant) or be exercised via `code/train_tokenizer.py` on a pre-subsampled file for local dev — never by editing `generate_tokenizers.py`.

### Starting Vocab Size (CORE-01)
- **D-05:** Phase-1 starting `vocab_size ≈ 8000` (closer to the research ~8k–16k F1 sweet spot than the scaffold's 5000, still fast on a subsample, comfortably clears the 0.4 floor). This is a starting point only — **Phase 3 sweeps vocab size per domain**, so Phase 1 must not hard-code assumptions that block a sweep.

### Cross-Word Bigram Mechanics (CORE-02, CORE-03, CORE-04) — carried from PROJECT.md, reaffirmed
- **D-06:** Space marker is `▁` (U+2581), added to the vocab as a **first-class mergeable symbol**; `space_token` attribute set to it.
- **D-07:** Merges may cross the marker but are **hard-capped at one internal space** (filter any candidate pair whose merged surface would contain ≥2 markers, in BOTH train and encode). Guarantee ≥1 bigram deterministically: if no legal cross-space bigram emerges from normal training, **force the highest-count legal cross-space merge** (disqualification insurance).
- **D-08:** Validate the bigram gate the assignment-specific way early: a token is a bigram iff `tok.replace('▁',' ').strip()` contains an internal space; assert ≥1 such token and assert **no** token has 2+ internal spaces. (Mirror `check_submission.py`'s predicate exactly.)

### Decode / NER Alignment Contract (ALIGN-01..03)
- **D-09:** `decode` is a pure, length-preserving substitution: ids→tokens→concatenate→replace `▁` with a single space. **No** stripping, case-folding, NFC/normalization, or any transform that changes character count. Special tokens decode to empty/identity in a way that does not corrupt cumulative length.
- **D-10:** Build a **local alignment harness** (own script under `code/`, e.g. `test_tokenizer.py` additions or a new helper) that asserts, on dev_1/dev_2 samples: cumulative `decode(ids[:i+1])` lengths are monotonic and reconstruct exact char offsets, and `decode(encode(x)).replace(' ','') == x.replace(' ','')`. This must be green locally before any VM run.

### Serialization & Reproducibility (SERDE-01..03)
- **D-11:** All learned state in plain picklable containers (`dict`/`list`/`tuple`/`str`/`int`): merge ranks `{(str,str):int}`, vocab maps, `space_token`. **No** lambdas / closures / `defaultdict(lambda)` / local classes. Convert any `defaultdict` to plain `dict` before pickling.
- **D-12:** Store the regex as a **pattern string** (compile in `__init__`/`__setstate__`); drop the runtime encode cache via `__getstate__`/`__setstate__` and rebuild on load. Class stays in `code/bpe_tokenizer.py` so it unpickles from `code/` alone.

### Claude's Discretion
- **Whitespace / decode edges** (user deferred): I'll preserve exact characters; emit one `▁` per literal space; decide multi-space/tab/newline handling empirically from the actual dev data so encode/decode stays length-symmetric (the NER `text` is single-space-joined via `' '.join(words)`, so the measured path is single-space — I'll confirm against `domain_*_dev.txt` raw lines too).
- **Pre-tokenization regex** pattern (GPT-2-like, space-aware, Twitter-tolerant for URLs/@/#) — exact pattern is mine to choose; it must keep the marker as a real emitted symbol so cross-space merges remain legal.
- **Merge tie-breaking rule** (for deterministic reproducibility) — I'll pick a deterministic rule (e.g. lexicographic on the pair) and document it.
- **Exact Phase-1 subsample size** and the bigram-forcing threshold — my discretion, tuned to keep local iteration fast while clearing F1≥0.4.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning
- `.planning/PROJECT.md` — constraints, key decisions (space marker, cross-word merges, length-faithful decode, pickle-safety), domains, grading.
- `.planning/REQUIREMENTS.md` — Phase-1 REQ-IDs (CORE-01..05, ALIGN-01..03, SERDE-01..03, EVAL-01, PKG-01) and out-of-scope.
- `.planning/ROADMAP.md` §"Phase 1" — goal + 5 success criteria.

### Research (algorithm/design guidance)
- `.planning/research/STACK.md` — BPE train/encode algorithms, merge-rank encode, pickle-safety patterns, vocab trade-off.
- `.planning/research/ARCHITECTURE.md` — decode length-faithfulness contract, component boundaries vs fixed scaffold, build order, local-vs-VM split.
- `.planning/research/PITFALLS.md` — disqualification gates (priority order), detection + prevention per pitfall.
- `.planning/research/FEATURES.md` — table-stakes vs differentiators; byte-fallback rationale.
- `.planning/research/SUMMARY.md` — synthesized roadmap implications.

### Fixed scaffold (PROVIDED — do NOT edit `base_tokenizer.py`, `train_ner_model.py`, `generate_tokenizers.py`, `check_submission.py`)
> Source: `code_env_example.zip` (to be extracted into the project as the first Phase-1 task). Data source: `data (3).zip` → `data/` (domain_*_train/dev.txt, ner_data/*.tagged).
- `code/base_tokenizer.py` — abstract base: `__init__` special_tokens `{[PAD]:0,[UNK]:1,[BOS]:2,[EOS]:3}`, abstract `train/encode/decode`, `save`/`load` (pickle), `get_vocab_size`. Our class MUST satisfy this contract.
- `code/bpe_tokenizer.py` — the stub we implement (the ONE submitted module).
- `generate_tokenizers.py` — reproducibility entrypoint; adds `code/` to path, imports `BPETokenizer`, trains `tokenizer_{1,2,3}.pkl` from `--train_files_{1,2,3}`/`--vocab_size`/`--seed=42`. UNEDITED.
- `check_submission.py` — the acceptance gate; structure check + `space_token`/bigram check + NER smoke. UNEDITED.
- `code/train_ner_model.py` — the FIXED NER pipeline; `NERDataset` aligns tokens→words via `decode([id])` + cumulative `decode(ids[:i+1])` char spans, first-subtoken labeling (rest=-100). UNEDITED. Read to understand the decode contract; only file-path args change.
- `code/test_tokenizer.py` — efficiency (tokens/char), speed (tokens/sec), reconstruction harness. May be edited/extended (and submitted).
- `code/train_tokenizer.py` — single-tokenizer train/save helper. May be edited/extended.
- `README.md` — usage and submission expectations.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BaseTokenizer` (`code/base_tokenizer.py`): provides `save`/`load` (pickle), `encode_batch`/`decode_batch`, `get_vocab_size`, and the `token_to_id`/`id_to_token`/`special_tokens` maps initialized in `__init__`. Subclass and reuse — do not reimplement persistence.
- `code/test_tokenizer.py`: ready-made efficiency/speed/reconstruction measurement we can extend for the local alignment harness (D-10).
- `code/train_tokenizer.py`: ready-made single-tokenizer train+save CLI for local dev on a subsampled file (D-03).

### Established Patterns (constraints from the fixed scaffold)
- **Special tokens occupy ids 0–3** (`[PAD]`/`[UNK]`/`[BOS]`/`[EOS]`); our learned tokens must be added with ids ≥ 4 and never collide.
- **NER alignment depends entirely on decode lengths** — `train_ner_model.py` derives `token_char_spans` from cumulative `decode` length deltas vs the original `text`. This is the single load-bearing invariant (D-09).
- **`space_token` must be in `token_to_id`** for `train_ner_model.py` to set `space_token_id` (it checks `tokenizer.space_token in tokenizer.token_to_id`).
- **Pickle reload happens with only `code/` on the path** (check_submission copies `code/*.py` to root and imports). Keep all referenced helpers inside `code/`.

### Integration Points
- `generate_tokenizers.py` instantiates `BPETokenizer(vocab_size=args.vocab_size)` and calls `.train(texts)` then `.save(path)` — our `__init__` signature and `train`/`save` must match (vocab_size kwarg already in the stub).
- The submission zip layout (from the sample `HW2_123456789.zip`): `report_<id>.pdf`, `trained_tokenizers/tokenizer_{1,2,3}.pkl`, `code/` (our `bpe_tokenizer.py` + edited helpers only), `train_commands.txt` at root.

</code_context>

<specifics>
## Specific Ideas

- Mirror the provided environment exactly in the project root (`code/`, `data/`, root `generate_tokenizers.py`/`check_submission.py`, `pyproject.toml`, `uv.lock`, `init.sh`) so `generate_tokenizers.py` and `check_submission.py` run unmodified — extracting the two zips is the first Phase-1 task.
- The Phase-1 "submittable" zip is a real fallback: even though tokenizers 2 & 3 are refined later, Phase 1 should produce all three `tokenizer_{*}.pkl` (using the default `generate_tokenizers.py` train-file mapping) so `check_submission.py` passes end-to-end.
- A read-only copy of the scaffold already exists in the session scratchpad (already studied) — but the canonical project copy must come from extracting `code_env_example.zip`.

</specifics>

<deferred>
## Deferred Ideas

- **Encode-speed / training-efficiency** (rank-map greedy + per-chunk cache + incremental inverted-index training on the full 95MB) → **Phase 2** (SPEED-01..03). Phase 1 may use a simple correct trainer/encoder.
- **Per-domain vocab sweep & tuning** for tokenizers 1 & 2 → **Phase 3** (TUNE-01..04).
- **Tokenizer-3 data-mix / dedup / generalization** → **Phase 4** (GEN-01..03).
- **Twitter-specific preprocessing** (URLs/@/#/emoji) as an efficiency/F1 lever → v2 OPT-03, only if it helps without breaking decode faithfulness.

None of the above are implemented in Phase 1 — discussion stayed within phase scope.

</deferred>

---

*Phase: 1-Gates-First Submittable Tokenizer*
*Context gathered: 2026-06-24*
