# Roadmap: BPE Tokenizer for Cross-Domain NER (NLP HW2)

## Overview

This roadmap delivers three correct, reproducible from-scratch BPE tokenizers that feed a fixed binary NER model, packaged as `HW2_214034621.zip`. The journey is **gates-first, compete-second**: Phase 1 ships a complete-but-minimal tokenizer that already passes every hard disqualification gate in `check_submission.py` (char-up BPE, mergeable space marker, >=1 cross-word bigram, no trigram, valid `space_token`, pickle-reloadable, deterministic, correct zip structure) and clears the F1>=0.4 floor end-to-end on one domain — a genuinely submittable artifact. Each later phase thickens one competition axis (fast encode + efficient training, per-domain vocab tuning, hidden-domain generalization, full VM evaluation/measurement) and ends with report + final packaging, leaving a still-submittable artifact at every boundary. Correctness is locked locally on CPU before any GPU spend to protect the 10-hour Tesla M60 VM budget.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Gates-First Submittable Tokenizer** - Complete minimal BPE that passes all hard gates, decodes length-faithfully, pickles/reproduces, and clears F1>=0.4 on one domain end-to-end
- [ ] **Phase 2: Fast Encode + Training Efficiency** - Rank-map greedy encoder with per-chunk cache and weighted/incremental training so the corpora train fast and the speed axis is competitive
- [ ] **Phase 3: Per-Domain Tuning (Tokenizers 1 & 2)** - Vocab-size sweep on the F1/efficiency knee to tune Twitter (tok 1) and news (tok 2) configs
- [ ] **Phase 4: Tokenizer 3 Hidden-Domain Generalization** - Balanced/deduped domain-1+2 mix + byte fallback so the unseen domain incurs zero UNK and degrades gracefully
- [ ] **Phase 5: VM NER Evaluation & Measurement** - All three tokenizers measured for F1, efficiency, and speed on the VM within the 10h budget, with `train_commands.txt` finalized
- [ ] **Phase 6: Report & Final Packaging** - <=1-page formatted report with all mandatory content and the final checker-validated submission zip

## Phase Details

### Phase 1: Gates-First Submittable Tokenizer
**Goal**: A complete, minimal `code/bpe_tokenizer.py` that passes every hard disqualification gate, decodes length-faithfully for NER alignment, pickles/reproduces from `code/`, and clears the F1>=0.4 floor on one domain — a genuinely submittable `HW2_214034621.zip`.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, ALIGN-01, ALIGN-02, ALIGN-03, SERDE-01, SERDE-02, SERDE-03, EVAL-01, PKG-01
**Success Criteria** (what must be TRUE):
  1. `check_submission.py` runs on the built zip and reports no errors, including >=1 bigram token and no 3+-word token (no internal-space-count > 1).
  2. The tokenizer's `space_token` is set non-None, present in the vocab, and a fresh-process `BaseTokenizer.load` of `tokenizer_1.pkl` from `code/` alone succeeds.
  3. A local alignment harness shows `decode` is length-faithful (cumulative-prefix lengths monotonic, exact char reconstruction on dev_1 and dev_2 samples) and `decode(encode(text))` reconstructs text per the checker's test.
  4. Running the unedited `generate_tokenizers.py` twice produces byte-identical `tokenizer_1.pkl` (deterministic/reproducible), with zero `[UNK]` on byte/char-level fallback inputs.
  5. NER F1 on `dev_1_binary.tagged` using tokenizer 1 is >= 0.4 end-to-end.
**Plans**: TBD

### Phase 2: Fast Encode + Training Efficiency
**Goal**: Replace any naive paths with a tiktoken-style rank-map greedy encoder plus per-chunk cache and weighted-frequency/incremental-index training, so encode is fast (competition speed axis) and the 95MB/29MB corpora train in reasonable time — output identical to the slow correct oracle.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: SPEED-01, SPEED-02, SPEED-03
**Success Criteria** (what must be TRUE):
  1. `encode` uses a merge-rank dict (greedy lowest-rank apply), not an O(vocab x n) whole-string rescan, and its output is byte-identical to the Phase 1 slow encoder on dev samples (equality oracle passes).
  2. A per-chunk encode cache memoizes repeated segments and measurably reduces encode time on Zipfian dev text versus an uncached run.
  3. Training uses weighted word-frequency chunks + incremental pair-count deltas with an inverted `pair->chunks` index (no full recount per merge), and trains tokenizer 1 on `domain_1_train.txt` to completion in reasonable wall time without OOM.
  4. All Phase 1 gates still pass after the rewrite (`check_submission.py` clean, fresh-process load OK, byte-reproducible) — the artifact remains submittable.
**Plans**: TBD

### Phase 3: Per-Domain Tuning (Tokenizers 1 & 2)
**Goal**: Tune tokenizer 1 (Twitter, `domain_1_train.txt`) and tokenizer 2 (news, `domain_2_train.txt`) by sweeping vocab size and scoring the three axes on dev, choosing configs that sit on the F1/efficiency knee without wrecking entity-boundary alignment.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: TUNE-01, TUNE-02, TUNE-03, TUNE-04
**Success Criteria** (what must be TRUE):
  1. Tokenizer 1 is trained on `domain_1_train.txt` and tokenizer 2 on `domain_2_train.txt`, each with a recorded vocab-size choice.
  2. A dev-set evaluation harness reports tokens/char efficiency and tokens/sec speed across multiple vocab sizes for each domain, producing a sweep table.
  3. The chosen per-domain configs balance efficiency against alignment (no over-merge that straddles entity boundaries), and both tokenizers still pass all Phase 1 gates and the encode equality oracle.
**Plans**: TBD

### Phase 4: Tokenizer 3 Hidden-Domain Generalization
**Goal**: Build tokenizer 3 on a deliberate, balanced, deduped mix of domain-1+2 data with a moderate vocab and byte fallback so an unseen hidden domain incurs zero UNK and degrades gracefully, plus a reasoned evidence-backed expectation of its behavior for the report.
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: GEN-01, GEN-02, GEN-03
**Success Criteria** (what must be TRUE):
  1. Tokenizer 3 trains on a documented mix of domain 1+2 data that downsamples the larger domain and deduplicates near-duplicate lines (the `--train_files_3` recipe is recorded).
  2. An out-of-domain / OOV stress test confirms tokenizer 3 produces zero `[UNK]` on unseen-character and held-out inputs via byte fallback.
  3. A reasoned, evidence-backed evaluation of tokenizer 3's expected hidden-domain efficiency/F1 behavior is written up for the report, and tokenizer 3 passes all Phase 1 gates.
**Plans**: TBD

### Phase 5: VM NER Evaluation & Measurement
**Goal**: With all local gates green, run NER on the course VM to confirm the F1>=0.4 floor on domain 2 and capture the competition numbers (F1, efficiency, speed) for all three tokenizers within the 10h budget, and freeze `train_commands.txt`.
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: EVAL-02, EVAL-03, EVAL-04, SERDE-04
**Success Criteria** (what must be TRUE):
  1. NER F1 on `dev_2_binary.tagged` using tokenizer 2 is >= 0.4 (domain-2 floor cleared on the VM).
  2. Tokenizer efficiency (tokens/char or tokens/sentence) and tokenization speed are measured and logged on the dev files for all three tokenizers.
  3. All GPU NER runs complete on the course VM with `time_left` showing budget remaining (within 10h), executed only after local gates were green.
  4. `train_commands.txt` at the zip root lists the exact commands (vocab sizes, `--train_files_3`, etc.) that reproduce `tokenizer_{1,2,3}.pkl`.
**Plans**: TBD

### Phase 6: Report & Final Packaging
**Goal**: Produce the <=1-page formatted report with all mandatory content and assemble the final `HW2_214034621.zip` that passes `check_submission.py` with all forbidden files byte-identical to the originals.
**Mode:** mvp
**Depends on**: Phase 5
**Requirements**: PKG-02, PKG-03
**Success Criteria** (what must be TRUE):
  1. `report_214034621.pdf` meets every format rule: <=1 A4 page, Arial 10, 1.15 spacing, 2.54cm margins, single column, captioned centered images, in-image text >=9pt.
  2. The report contains all mandatory content: author (Diyar Husayyan, 214034621); per-tokenizer deviation-from-basic-BPE + hidden-domain handling; top-5 & bottom-5 training bigrams; per-model dev F1 + efficiency; hidden-domain evaluation; model-2 performance + parameter/characteristic/improvement discussion.
  3. The final `HW2_214034621.zip` passes `check_submission.py` with no errors, the required structure exactly, and `base_tokenizer.py`/`train_ner_model.py`/`generate_tokenizers.py`/`check_submission.py` byte-identical to the provided originals.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Gates-First Submittable Tokenizer | 0/TBD | Not started | - |
| 2. Fast Encode + Training Efficiency | 0/TBD | Not started | - |
| 3. Per-Domain Tuning (Tokenizers 1 & 2) | 0/TBD | Not started | - |
| 4. Tokenizer 3 Hidden-Domain Generalization | 0/TBD | Not started | - |
| 5. VM NER Evaluation & Measurement | 0/TBD | Not started | - |
| 6. Report & Final Packaging | 0/TBD | Not started | - |
