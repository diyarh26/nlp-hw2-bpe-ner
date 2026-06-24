# Project Research Summary

**Project:** BPE Tokenizer for Cross-Domain NER (NLP HW2)
**Domain:** From-scratch Byte Pair Encoding tokenizer (pure Python/numpy) feeding a fixed binary NER model; three tokenizers (Twitter, news, hidden domain)
**Researched:** 2026-06-24
**Confidence:** HIGH on gates, algorithms, and architecture; MEDIUM-HIGH on the assignment-specific bigram adaptation and competition tuning magnitudes

## Executive Summary

This is a single-module algorithmic engineering problem disguised as an ML task. You own exactly one file - `code/bpe_tokenizer.py`, a `BaseTokenizer` subclass implementing `train`/`encode`/`decode` - that plugs into a fixed, graded-against-original NER scaffold. Experts build BPE the way tiktoken and SentencePiece do: a weighted word-frequency table for training, incremental pair-count deltas with an inverted (pair to chunk) index for the merge loop, and a greedy lowest-merge-rank encoder with a per-chunk cache. The deliberate deviation from textbook BPE is the whole game: the space is a **mergeable symbol** (SentencePiece marker U+2581), so merges legally cross exactly one space to produce the mandatory cross-word **bigram** token - never two.

The grade splits into a 20% correctness/F1>=0.4 gate, a 45% three-axis competition (token efficiency, encode speed, F1 - equally weighted and in tension), 15% hidden-domain analysis, and a 20% report. The recommended approach is **gates first, compete second**: lock every disqualification gate (bigram present, no trigram, length-faithful decode, `space_token` valid, pickle-safe, deterministic/reproducible) before touching the efficiency/speed/F1 frontier. Token efficiency and F1 actively pull against each other - aggressive cross-word merges cut tokens but a bigram straddling an entity boundary collapses first-subtoken labeling - so vocab size and merge aggressiveness must be **tuned on the dev F1/efficiency knee**, not maximized. For the hidden domain, a **byte-level base + byte fallback** (zero UNK) plus a balanced, deduped domain-1+2 data mix is the single highest-leverage move.

The dominant risks are silent and gate-shaped. A non-length-faithful decode passes a naive round-trip test but drifts the NER aligner's cumulative char offsets and quietly craters F1. An unpicklable lambda/compiled-regex on `self` fails `BaseTokenizer.load` and zeroes a tokenizer. Non-deterministic tie-breaking breaks byte-reproducibility from `generate_tokenizers.py`. And the 10-hour Tesla M60 VM budget can be burned discovering any of these on the VM - when every one of them is testable locally on CPU. Mitigation is a hard "all local gates green before any VM run" discipline backed by automated assertions.

## Key Findings

### Recommended Stack

The "stack" is an algorithm + data-structure stack, not packages - the library set is fixed (numpy, regex, torch, tqdm + stdlib). Training uses a deduped word-frequency table so cost scales with unique chunks, not corpus size, combined with incremental pair-count deltas and an inverted pair-to-chunk index (the ~100-2000x speedup over naive recount). Encode is tiktoken-style: a merge-rank dict, greedily applying the lowest-rank adjacent pair per pre-tokenized chunk, with a plain-dict per-chunk cache that dominates real-world speed on Zipfian text. See `.planning/research/STACK.md`.

**Core technologies:**
- **Weighted word-frequency dict + symbol-list chunks** - TRAINING corpus representation - collapses the 95MB corpus to a few hundred K unique weighted chunks; the single biggest training win.
- **Incremental pair-count deltas + inverted index** - TRAINING merge loop - only re-touch chunks containing the merged pair; turns hours into seconds in pure Python.
- **Merge-rank dict + greedy lowest-rank apply** - ENCODE - exactly tiktoken's reference encoder; deterministic, matches training, plain-dict picklable.
- **Per-chunk encode cache (plain dict)** - ENCODE - memoize chunk to ids; often the largest real speedup; must pickle cleanly (no `lru_cache`).
- **`regex` pre-tokenization with explicit space marker** - TRAIN+ENCODE - bounds inner merge loops to short chunks and makes spaces first-class mergeable symbols.

### Expected Features

Gate features are dictated by `check_submission.py`, the decode-length alignment mechanism, and the F1 gate; differentiators win the 45% competition and 15% hidden-domain blocks. The three competition axes (efficiency, speed, F1) pull against each other and must be jointly tuned. See `.planning/research/FEATURES.md`.

**Must have (table stakes - missing = disqualification or zeroed score):**
- True char/byte-up BPE subclassing `BaseTokenizer` (`train`/`encode`/`decode`)
- >=1 cross-word bigram token (exactly one internal space) via mergeable space marker
- Hard cap at one internal space (reject any merge yielding 2+ markers)
- `space_token` set non-None and present in vocab
- Length-faithful, reversible decode (load-bearing for NER span alignment)
- Pickle-safe state (no lambdas/closures/compiled-regex on instance)
- Deterministic, reproducible via unedited `generate_tokenizers.py` + `train_commands.txt`
- F1 >= 0.4 on both dev sets

**Should have (competitive - the 45% + 15%):**
- Fast rank-map encode + per-segment cache (speed axis)
- Per-domain vocab-size sweep on the F1/efficiency knee (tokenizers 1 & 2)
- Byte-level base + byte fallback to zero UNK (highest-leverage hidden-domain feature; ~+0.87 F1 in literature on unseen social text)
- Tokenizer-3 balanced/deduped data mix + merge regularization for generalization
- Twitter URL/@/#/emoji length-preserving segmentation; news collocation/casing tuning

**Defer / sequence after gates:**
- Speed optimization and vocab sweeps come *after* correctness is locked
- Top-5/bottom-5 bigram extraction + per-model F1/efficiency tables (report deliverable)

**Explicit anti-features (do NOT build):** tokens spanning 3+ words; max-vocab greedy merging; UNK/char-dropping; length-altering normalization (lowercasing, NFC/NFKC, accent strip); GPT-2 whitespace pre-tokenization; naive O(n^2) re-encode; specializing tokenizer 3 to one domain; lambdas/compiled-regex on `self`; external data/libs.

### Architecture Approach

A self-contained subclass inside a fixed scaffold - the only owned logic is `code/bpe_tokenizer.py` (plus optional helper scripts in `code/`). The load-bearing data flow is encode to NER alignment: `train_ner_model.py` reconstructs token-to-word char spans purely from `decode([id])` and cumulative `decode(ids[:i+1])` lengths versus the original text, labeling only the first subtoken per word (rest = -100). The "scaling" axis is not users but corpus size (95MB+29MB) and the 10h GPU budget; everything except final NER F1 runs on local CPU. See `.planning/research/ARCHITECTURE.md`.

**Major components:**
1. `train(corpus)` - learn merge table from chars upward over a space-marked symbol stream; build vocab/id maps atop fixed specials (PAD0/UNK1/BOS2/EOS3); set `space_token`; guarantee >=1 bigram; cap at one internal space.
2. Merge table + vocab - ordered merges `(a,b)->ab` with rank; plain dicts/lists/strings (picklable).
3. `encode(text)` - pre-tokenize with same regex+marker rule; greedy lowest-rank merge per chunk (heap/linked-list to avoid O(n^2)); per-chunk cache.
4. `decode(ids)` - pure length-faithful join + marker-to-space substitution, excluding specials; the entire NER alignment contract lives here.

### Critical Pitfalls

Top disqualification/gate risks (full list of 12 with recovery costs in `.planning/research/PITFALLS.md`):

1. **[DQ] GPT-2 pre-tokenization to zero bigrams** - make the space a mergeable marker symbol; assert the checker's own predicate inside `train`.
2. **[DQ] Trigram leakage (2+ internal spaces)** - add a merge-eligibility filter rejecting any pair whose merged token has >1 internal space, in both `train` and `encode`; post-train vocab assertion.
3. **[DQ/GATE] Non-length-faithful decode** - silently misaligns NER labels and tanks F1 with no error. Test cumulative-prefix length monotonicity + exact char reconstruction over real dev sentences, not just round-trip identity.
4. **[DQ] `space_token` None / absent / colliding** - single rare codepoint (U+2581) guaranteed absent from corpus; assert in vocab and no collision.
5. **[DQ/REPRO] Pickle won't reload from `code/` alone** - no lambdas/closures/compiled-regex/renamed class; fresh-process load test after every change.
6. **[REPRO] Non-deterministic training** - total deterministic tie-break (lexicographic / first-occurrence), seed everything, sort before iterating; double-run byte-diff as acceptance test.
7. **[COMP/GATE] Over-merging tanks F1** - tune vocab/merge count on the F1/efficiency knee; bigrams that glue capitalized words to neighbors straddle entity boundaries.
8. **[GATE/COMP] UNK explosion on hidden domain** - byte/char fallback covering all 256 bytes; OOV stress test.
9. **[COMP] Naive O(n^2) encode loses speed axis** - rank/heap or linked-list encoder, with the slow rescan kept only as an equality oracle.
10. **[REPRO/COMP] Burning the 10h VM budget on locally-findable bugs** - "local gates pass" precondition before any VM run.
11. **[DQ/REPRO] Editing forbidden files / wrong zip structure** - keep forbidden files byte-identical; scripted packaging; `check_submission.py` as final gate.
12. **[COMP] Twitter junk merges / OOM on 95MB** - word-freq table + streaming + incremental counts; deliberate length-preserving Twitter handling.

## Implications for Roadmap

Based on combined research, the natural build order is **gates -> competition -> generalization -> packaging**, with correctness fully locked locally before any GPU spend. Suggested phase structure:

### Phase 1: Core BPE merge engine (char/byte-up, mergeable space, bigram-capped)
**Rationale:** This is THE foundational design decision (Pitfalls 1, 2, 4); getting space-as-mergeable-symbol and the <=1-space cap wrong means an algorithm rewrite. Everything depends on it.
**Delivers:** `train()` producing a merge table + vocab/id maps atop the 4 fixed specials; `space_token` set; >=1 bigram guaranteed; trigram-leakage filter; byte/char base alphabet (no UNK).
**Addresses:** char-up BPE, cross-word bigram, <=1-space cap, byte fallback, `space_token` (FEATURES table stakes).
**Avoids:** Pitfalls 1 (zero bigram), 2 (trigram), 4 (space_token), 8 (UNK foundation).

### Phase 2: Decode/alignment contract + correctness harness
**Rationale:** Decode length-faithfulness gates everything F1-related and fails silently (Pitfall 3) - it is the highest-value local test in the project and must be trustworthy before any tuning or VM run.
**Delivers:** length-faithful `decode`; the cumulative-prefix monotonicity + exact-reconstruction invariant test (ARCHITECTURE `assert_length_faithful`); reconstruction test per `test_tokenizer.py` rule.
**Implements:** the encode-to-NER alignment data flow.
**Avoids:** Pitfall 3 (silent misalignment).

### Phase 3: Serialization + determinism/reproducibility
**Rationale:** Pickle-load and byte-reproducibility are gates that break on refactor and must become standing checks (Pitfalls 5, 6) before further iteration accumulates risk.
**Delivers:** fresh-process `BaseTokenizer.load` test; banished lambdas/compiled-regex; total deterministic tie-break + seeding; `generate_tokenizers.py` double-run byte-diff.
**Avoids:** Pitfalls 5 (pickle load), 6 (non-determinism).

### Phase 4: Training efficiency + fast encode
**Rationale:** The 95MB corpus and the timed encode axis demand the weighted-table/incremental-index training and rank-map encoder; build the correct slow path first as the equality oracle (Pitfalls 9, 12).
**Delivers:** weighted word-freq table + incremental pair-count deltas + inverted index; streaming file reads; rank/heap or linked-list encoder + per-chunk cache; benchmark vs naive oracle (equal output).
**Uses:** STACK incremental-delta training + tiktoken greedy encode + per-chunk cache.
**Avoids:** Pitfalls 9 (O(n^2) encode), 12 (OOM/slow train).

### Phase 5: Per-domain tuning (tokenizers 1 & 2) + hidden-domain generalization (tokenizer 3)
**Rationale:** Competition points (45%) and the 15% hidden-domain block live here, and require Phases 1-4 green plus a trustworthy alignment test before sweeping the F1/efficiency frontier (Pitfalls 7, 8).
**Delivers:** per-domain vocab sweep picking the F1/efficiency knee; Twitter length-preserving segmentation; news collocation/casing tuning; tokenizer-3 balanced/deduped 1+2 mix + merge regularization; chosen `--train_files_3` recipe; OOV stress test (zero UNK).
**Addresses:** all differentiator features.
**Avoids:** Pitfalls 7 (over-merge F1 collapse), 8 (hidden-domain UNK).

### Phase 6: VM NER evaluation (F1 gate + competition numbers)
**Rationale:** Only stage needing GPU; gated behind all local gates green to protect the 10h budget (Pitfall 10). Run the highest-risk tokenizer first to validate end-to-end once.
**Delivers:** F1 >= 0.4 on dev_1/dev_2; logged F1 + efficiency + speed per tokenizer; hidden-domain numbers.
**Avoids:** Pitfall 10 (budget burn).

### Phase 7: Report + packaging
**Rationale:** Strict format gates and the 20% report; `check_submission.py` on the actual zip is the final acceptance gate (Pitfall 11).
**Delivers:** <=1-page report (top/bottom-5 bigrams, per-model F1+efficiency, hidden-domain + model-2 discussion); `train_commands.txt`; scripted `HW2_214034621.zip`; forbidden-files byte-diff; final `check_submission.py`.
**Avoids:** Pitfall 11 (forbidden edits / bad zip).

### Phase Ordering Rationale

- **Dependency-driven:** the merge engine (1) must exist before decode (2); decode correctness (2) must be trustworthy before pickle/determinism (3) and before any tuning (5) or VM run (6). Fast paths (4) need the slow correct path as an oracle.
- **Architecture-driven:** the encode-to-decode-to-NER-alignment flow makes decode the linchpin, so it gets its own early phase rather than being folded into encode.
- **Pitfall-driven:** all disqualification gates (1-6, 11) are front-loaded to local CPU; the GPU phase (6) is deliberately last and gated, mapping directly to the "local gates pass" discipline.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 5 (tuning + generalization):** the F1/efficiency knee, Twitter length-preserving normalization choices, tokenizer-3 mix ratio and merge regularization are assignment-specific with no off-the-shelf reference (MEDIUM confidence) - empirical sweeps, not lookups.
- **Phase 1 (bigram adaptation specifics):** the exact mergeable-space mechanics and trigram filter are reasoned from the spec, not a reference implementation - worth validating against `check_submission.py`'s actual predicate early.

Phases with standard, well-documented patterns (skip research-phase):
- **Phase 4 (training efficiency + fast encode):** weighted-table/incremental-index training and rank-map greedy encode are HIGH-confidence, corroborated by tiktoken/HF/SentencePiece.
- **Phase 3 (pickle-safety + determinism):** established stdlib semantics.
- **Phase 7 (packaging):** mechanical, driven entirely by `check_submission.py` and the sample zip.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core train/encode/pickle patterns verified against tiktoken, HF course, Sennrich; bigram adaptation MEDIUM-HIGH (no external reference). |
| Features | MEDIUM-HIGH | Gate features HIGH (anchored in PROJECT.md/`check_submission.py`); competition/generalization tactics MEDIUM (BPE/SentencePiece literature). |
| Architecture | HIGH | Scaffold contracts fully specified in PROJECT.md; SentencePiece marker + decode-length alignment corroborated; O(n log n) encode MEDIUM-HIGH. |
| Pitfalls | HIGH | Derived directly from spec mechanisms; trade-off magnitudes MEDIUM. |

**Overall confidence:** HIGH on what must be true to pass; MEDIUM-HIGH on what wins competition points.

### Gaps to Address

- **Bigram-merge mechanics have no reference implementation** - validate the mergeable-space + <=1-space filter against `check_submission.py`'s exact predicate in Phase 1, not at packaging.
- **F1/efficiency knee and vocab sizes per domain are empirical** - must be swept on dev in Phase 5; no a-priori "correct" value. Budget VM runs accordingly.
- **Tokenizer-3 data mix ratio and merge-regularization strength are heuristic** - proxy-test with a held-out OOV slice locally before committing VM time.
- **Multi-space / tab / newline handling** in encode must be deliberately mirrored in decode (length-faithfulness corner case) - decide in Phase 2.
- **Quantified efficiency-vs-F1 trade-off magnitudes are MEDIUM** - treat literature figures as directional, confirm on dev.

## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md` - gate spec, NER decode-length alignment, bigram predicate, reproducibility, 10h VM budget, data sizes, forbidden files.
- tiktoken `_educational.py` / DeepWiki - greedy lowest-merge-rank encode, list-of-parts representation.
- HuggingFace LLM Course ch.6 - canonical BPE train/encode reference.
- Sennrich et al. 2016 (arXiv 1508.07909) - original BPE-for-NLP algorithm.
- SentencePiece (Kudo & Richardson, arXiv 1808.06226 + repo) - U+2581 marker, lossless length-faithful decode.

### Secondary (MEDIUM confidence)
- "From Hours to Seconds: Optimising BPE Tokeniser Training" (Thomson, Medium) - weighted-chunk dedup, pair deltas, inverted index (~2000x).
- Rust NLP tales / "A Formal Perspective on BPE" (arXiv 2306.16837) - linked-list + priority-queue O(n log n) encode.
- SMILE (arXiv 2307.00135) - byte fallback +0.87 F1 on unseen social-media domain.
- HF token classification / arXiv 1809.08386 - first-subtoken labeling & boundary alignment.
- Data Mixture Inference (arXiv 2407.16607), NeurIPS 2024 BPE-training-data paper - merge-mix and tie-break determinism.

### Tertiary (LOW confidence)
- Assignment-specific bigram cross-word adaptation, vocab/F1 trade-off magnitudes, tokenizer-3 mix recipe - reasoned from spec; validate empirically during Phases 1 and 5.

---
*Research completed: 2026-06-24*
*Ready for roadmap: yes*
