# Pitfalls Research

**Domain:** From-scratch BPE tokenizer feeding a fixed binary NER model; 3 tokenizers (Twitter, news, hidden); judged on token efficiency, encode speed, F1, plus strict format/reproducibility gates.
**Researched:** 2026-06-24
**Confidence:** HIGH (most pitfalls derive directly from the spec mechanisms in PROJECT.md and `check_submission.py`/`train_ner_model.py` behavior; trade-off magnitudes are MEDIUM)

> Priority key: **[DQ]** = disqualification / grade-0 gate (highest priority). **[GATE]** = fails the 20% F1≥0.4 implementation gate. **[COMP]** = costs competition points (45%) but not disqualifying. **[REPRO]** = breaks reproducibility/packaging.

## Critical Pitfalls

### Pitfall 1: [DQ] GPT-2-style pre-tokenization → zero bigram tokens

**What goes wrong:**
You implement "textbook" BPE that splits on whitespace first (GPT-2 / standard pre-tokenization) so merges never cross a space. `check_submission.py` finds no token where `tok.replace(space_token,' ').strip()` contains an internal space → **no bigram token → disqualification**.

**Why it happens:**
Every tutorial and reference BPE (Karpathy minBPE, HF tutorials, the original Sennrich code) pre-tokenizes on whitespace because cross-word merges are normally undesirable. Copying any of them verbatim guarantees zero bigrams. The constraint here is the *opposite* of the industry default.

**How to avoid:**
Design the merge loop to operate over a stream where the space is a regular, **mergeable** symbol (SentencePiece-style `▁` marker, or keep raw space as a vocab symbol). A pair like `(the, ▁York)` must be eligible to merge. Verify a bigram exists by replicating the checker's exact predicate on your own vocab: `any(t.replace(space_token,' ').strip().find(' ')>=0 for t in vocab)`. Make this assertion part of `train`.

**Warning signs:**
No vocab token contains the space marker in a word-internal position; running your own copy of the checker's bigram test returns False; token-count efficiency barely beats whitespace tokenization.

**Phase to address:**
Core BPE design phase (the merge algorithm). This is THE foundational design decision — wrong here means a rewrite.

---

### Pitfall 2: [DQ] Trigram leakage — merges produce tokens with 2+ internal spaces

**What goes wrong:**
Once cross-word merges are allowed, BPE will happily merge a bigram token with an adjacent word (e.g. `(of▁the) + ▁York → of▁the▁York`). That token has two internal spaces → spans 3 words → violates the bigram ceiling → **disqualification**.

**Why it happens:**
BPE is greedy and recursive: any frequent pair gets merged, including pairs where one or both sides already contain a space marker. Nothing in vanilla BPE caps the "word count" of a merged symbol. The constraint (exactly ≤1 internal space) is a custom guard you must add.

**How to avoid:**
Add a **merge-eligibility filter** before accepting any merge: reject a candidate pair `(a,b)` if the resulting token would contain ≥2 internal spaces, i.e. `(a+b).replace(space_token,' ').strip().count(' ') > 1`. Apply this both during `train` (don't add the merge rule) and as belt-and-suspenders during `encode`. Add a post-train assertion scanning the full vocab for any token with >1 internal space.

**Warning signs:**
Vocab inspection shows tokens like `word▁word▁word`; the count of internal spaces in your longest tokens exceeds 1; `check_submission.py` flags a trigram.

**Phase to address:**
Same phase as Pitfall 1 (merge algorithm). The eligibility filter is the twin constraint to "allow cross-word merges."

---

### Pitfall 3: [DQ/GATE] Decode is not length-faithful → NER token→word alignment silently breaks

**What goes wrong:**
`train_ner_model.py` derives token→word char spans by calling `decode([id])` and cumulative `decode(token_ids[:i+1])`, comparing **string lengths** against the original text to find where each token lands. If your `decode` drops/adds characters, doesn't restore real spaces, or isn't a left-to-right concatenation whose cumulative length matches the source, the spans drift. Labels attach to the wrong words → garbage training signal → low/zero F1 (can drop below the 0.4 gate, and silently — no error is raised).

**Why it happens:**
SentencePiece-style markers (`▁`) must be converted back to real spaces in `decode`, and the *length* must match the original surface form exactly (including the leading-space convention). People test `encode(decode(x))==x` for round-trip identity but never test the *cumulative length monotonicity* that alignment actually depends on. The failure is invisible locally because the tokenizer "works."

**How to avoid:**
Treat decode-length-faithfulness as a first-class invariant. Write a local test that, for many sample sentences, mirrors the NER aligner: walk `i` from 1..n, compute `len(decode(ids[:i]))`, assert it is non-decreasing and that `decode(all_ids)` reconstructs the original text character-for-character (modulo the documented normalization). Confirm the leading-space convention matches what the aligner expects. Do this BEFORE spending any VM time.

**Warning signs:**
`decode(encode(s)) != s` for some sentences; cumulative decode length jumps non-monotonically or overshoots the source length; first-subtoken labels land mid-word in a spot check; F1 is suspiciously near-random (~entity base rate) despite a reasonable tokenizer.

**Phase to address:**
Decode/alignment-contract phase — immediately after the merge algorithm, before any NER run. Highest-value local test in the project.

---

### Pitfall 4: [DQ] space_token is None / absent from vocab / wrong type

**What goes wrong:**
`space_token == None` is explicitly rejected. Also fatal: `space_token` set but not actually present as a symbol in the vocab, or it's a multi-char string that collides with normal text.

**Why it happens:**
The attribute is easy to forget to set during refactors, or it's set after pickling, or the marker chosen also appears in real input data (collision corrupts decode).

**How to avoid:**
Set `space_token` in `__init__`/`train` to a single rare codepoint (e.g. `▁` U+2581) guaranteed absent from training text; assert it's in the vocab after training; assert no raw training character equals the marker (pick a different one if collision). Cover with a unit test.

**Warning signs:**
`tokenizer.space_token is None`; marker char appears in raw corpus samples; decode produces stray marker glyphs.

**Phase to address:**
Core BPE design phase; verified in the local test-harness phase.

---

### Pitfall 5: [DQ/REPRO] Pickle won't reload from `code/` alone

**What goes wrong:**
The graded pipeline loads your tokenizer via `BaseTokenizer.load` (unpickle) using only the `code/` directory. If the pickle references a lambda, a closure, a `regex.Pattern` compiled object that won't round-trip, a method-bound default, or a class that was moved/renamed since pickling, **load fails → effectively grade 0** for that tokenizer.

**Why it happens:**
Convenience patterns sneak in: `defaultdict(lambda: ...)`, storing compiled regex objects, nested local classes, or `functools.partial` over a local. Re-pickling after a refactor that renamed the class also breaks old pickles. People pickle once early and don't re-test load after later edits.

**How to avoid:**
Ban unpicklable state: no lambdas (use module-level functions or plain dicts), no stored compiled-regex objects (store the pattern string and compile lazily in `__init__`/property), keep the class name/module path stable. After EVERY change to the tokenizer class, run a fresh-process load test: pickle, then in a NEW python process with cwd at `code/`, `BaseTokenizer.load(...)` and run `encode`/`decode`. Automate it.

**Warning signs:**
`PicklingError`/`AttributeError: Can't get attribute` on load; load works in-process but fails in a fresh interpreter; any `lambda` / `partial` / compiled-pattern attribute on `self`.

**Phase to address:**
Serialization phase, then re-verified in every subsequent phase that touches the class (make it a standing CI-style check).

---

### Pitfall 6: [REPRO] Non-deterministic training → not byte-reproducible from generate_tokenizers.py

**What goes wrong:**
The grader regenerates all 3 tokenizers from the unedited `generate_tokenizers.py` + `train_commands.txt`. If your output differs run-to-run, the regenerated tokenizer ≠ your submitted one → reproducibility failure, and your reported numbers become unverifiable.

**Why it happens:**
Hidden non-determinism: BPE pair-frequency **ties broken by dict/set iteration order**, `dict`/`Counter.most_common` order across versions, `set` ordering, multiprocessing/threadpool merge ordering, unseeded sampling when mixing domains for tokenizer 3, or reading large files in OS-dependent order. Python hash randomization can also perturb set/dict iteration for str keys in some constructs.

**How to avoid:**
Make tie-breaking **explicit and total**: when two pairs share max frequency, break ties by a deterministic key (e.g. lexicographic on the pair, or earliest first-occurrence) — never by raw dict order. Seed everything (Python `random`, numpy) from a fixed SEED. Sort before iterating where order matters. For tokenizer-3 domain mixing, use a seeded, deterministic sampler with the exact recipe encoded in `train_commands.txt`. Verify by running `generate_tokenizers.py` twice and diffing the pickles (or the merge lists) byte-for-byte.

**Warning signs:**
Two training runs produce different vocab/merge order; diff of two pickles is non-empty; F1/efficiency numbers wobble between runs with identical commands.

**Phase to address:**
Determinism/reproducibility phase (after merge algorithm, before final packaging). The double-run diff is the acceptance test.

---

### Pitfall 7: [COMP/GATE] Over-merging cross-word bigrams wrecks entity-boundary alignment and F1

**What goes wrong:**
Aggressive cross-word merging is great for the efficiency axis but corrosive for F1: when an entity word gets glued into a bigram token with a neighbor (e.g. `New▁York` as one token, or `President▁Obama` merged), first-subtoken labeling assigns ONE label to a token covering two words. The non-entity neighbor and the entity word now share a single label/first-subtoken — the model can't place the boundary correctly. Push merges too hard and efficiency rises while F1 collapses.

**Why it happens:**
The three axes are in tension and "compete hard" pressure pushes toward maximal merging. The efficiency win is immediately visible; the F1 cost is downstream on the VM and easy to under-weight. Cross-word bigrams are exactly the tokens most likely to straddle an entity boundary.

**How to avoid:**
Treat vocab size / merge count as a tuned hyperparameter on the efficiency↔F1 frontier, not "as big as possible." Sweep a few vocab sizes per domain and pick the knee, not the extreme. Consider biasing merges away from cross-word pairs that frequently straddle entity boundaries (though tokenizer is label-blind, common function-word+capitalized-word bigrams are prime offenders). Measure F1 AND efficiency at each setting; never optimize efficiency in isolation.

**Warning signs:**
Efficiency keeps improving but dev F1 drops as vocab/merges grow; many high-frequency bigrams glue a capitalized (likely-entity) word to a neighbor; F1 markedly worse than a whitespace-baseline tokenizer.

**Phase to address:**
Tuning/evaluation phase (per-domain vocab sweep). Requires the local alignment test (Pitfall 3) to be trustworthy first.

---

### Pitfall 8: [GATE/COMP] UNK explosion on the hidden domain (tokenizer 3)

**What goes wrong:**
Tokenizer 3 is trained on a mix of domains 1+2 but evaluated on an unseen domain 3. If it lacks a byte/char fallback, out-of-vocab characters/words become UNK. UNK tokens decode to a fixed string whose length usually ≠ the source word → breaks the decode-length alignment (Pitfall 3) AND destroys F1 on the very axis tokenizer 3 is graded on (15% hidden-domain analysis + competition F1).

**Why it happens:**
A pure word/subword BPE vocab can't represent unseen scripts, emoji, rare punctuation, or domain-specific tokens. People test only on dev_1/dev_2 (in-distribution) and never simulate an OOV domain, so the failure is invisible until the hidden eval.

**How to avoid:**
Guarantee total coverage with a **byte-level or full-character base alphabet** so every input is representable with zero true UNK — worst case it falls back to bytes/chars (length-faithful by construction). Choose a domain mix for tokenizer 3 that maximizes character/script coverage. Stress-test by holding out a slice of one domain (or injecting emoji/URLs/foreign chars) as a proxy "hidden" set and confirm zero UNK and stable alignment.

**Warning signs:**
Any UNK emitted on held-out/OOV text; vocab base alphabet doesn't cover all 256 bytes (or all chars seen in dev); tokenizer-3 efficiency much worse on OOV-looking input; decode length mismatches on OOV words.

**Phase to address:**
Tokenizer-3 / cross-domain generalization phase; byte fallback decided in core BPE design.

---

### Pitfall 9: [COMP] Naive O(n²) encode loses the speed axis

**What goes wrong:**
The reference BPE encode re-scans the whole token list after every merge (find best pair → merge → restart scan). On long sentences/large eval sets this is O(n²) and loses the speed axis (1/3 of the 45% competition).

**Why it happens:**
The simplest correct encoder is the rescan loop; it's what tutorials show. Speed only matters under the grader's timing harness, which you may not run until late.

**How to avoid:**
Use an efficient encoder: rank-based merges with a priority structure (heap of merge candidates with positions) or a linked-list/index approach so each merge is near-O(1) amortized, giving ~O(n log n) or O(n·#merges) bounded encode. Precompute a merge-rank dict for O(1) priority lookup. Benchmark encode locally on long Twitter lines BEFORE the VM. Keep correctness identical to the slow path (test equality of outputs).

**Warning signs:**
Encode time grows super-linearly with sentence length; profiling shows repeated full-list rescans; timing harness on VM is slow relative to token count.

**Phase to address:**
Encode-performance phase (after a correct slow encoder exists and is the equality oracle).

---

### Pitfall 10: [REPRO/COMP] Burning the 10-hour VM budget on bugs that were findable locally

**What goes wrong:**
The VM (Tesla M60) has a hard 10h total runtime budget for NER training/F1/timing across THREE tokenizers plus re-runs. Discovering alignment bugs, pickle-load failures, UNK issues, or non-determinism on the VM — things testable on CPU locally — can exhaust the budget before final scoring runs, leaving you unable to produce submission numbers.

**Why it happens:**
Tokenizer training is CPU-only and fast locally; the temptation is to "just run it on the VM and see F1." Each NER run is expensive, and three tokenizers × debugging iterations adds up fast.

**How to avoid:**
Front-load everything testable to local CPU: bigram/trigram assertions, decode-length alignment simulation, pickle fresh-process load, determinism double-run diff, UNK/OOV stress test, encode benchmark. Only go to the VM once a tokenizer passes ALL local gates. Budget VM time explicitly (e.g. ≤1 confirmatory run per tokenizer + buffer); check `time_left` before each run. Run the cheapest/highest-risk tokenizer first to validate the end-to-end pipeline once.

**Warning signs:**
You're debugging tokenizer logic on the VM; multiple failed NER runs per tokenizer; `time_left` dropping fast with no scored result yet.

**Phase to address:**
Cross-cutting / process discipline — enforced by a "local gates pass" checklist that is a precondition for every VM run.

---

### Pitfall 11: [DQ/REPRO] Editing forbidden files or wrong zip/file structure

**What goes wrong:**
Editing `base_tokenizer.py`, `train_ner_model.py`, `generate_tokenizers.py`, or `check_submission.py` (graded against originals), or producing a zip whose structure/filenames don't match exactly (`HW2_214034621.zip`, `report_214034621.pdf`, `train_commands.txt` at root, `code/` layout) → **grade 0**.

**Why it happens:**
You add a debug print to `train_ner_model.py`, or rename a path, or zip the wrong directory level (extra nesting), or misname by ID. Only file-path args in `train_ner_model.py` may legitimately differ per domain.

**How to avoid:**
Keep forbidden files read-only / under version control and diff against originals before packaging (assert byte-identical except allowed path args). Build the zip with a scripted, deterministic packaging step and run `check_submission.py` on it as the final gate. Compare your zip's tree against the provided `HW2_123456789.zip` sample exactly. Double-check the ID (214034621) in every filename.

**Warning signs:**
`git diff` shows changes in forbidden files; `check_submission.py` reports structure/filename errors; zip has an extra top-level folder; filenames contain the wrong ID.

**Phase to address:**
Packaging/submission phase; `check_submission.py` is the final acceptance gate.

---

### Pitfall 12: [COMP] Mishandling Twitter artifacts and OOM/slow training on the 95MB file

**What goes wrong:**
(a) Domain-1 Twitter text (`@mentions`, `#hashtags`, URLs, emoji, repeated chars, emoticons) produces a vocab full of one-off junk merges, hurting efficiency and generalization. (b) Loading/processing the 95MB (domain 1) and 29MB (domain 2) files naively (full read into memory, per-pair full-corpus recount each iteration) causes OOM or hours-long training on local CPU.

**Why it happens:**
Social text has heavy long-tail tokens; naive BPE training recomputes pair counts over the whole corpus every merge iteration (O(merges × corpus)). 95MB of one-sentence-per-line text is large for a pure-Python counter loop.

**How to avoid:**
Work on a **word-frequency representation**: collapse the corpus to a `{word: count}` (or symbol-sequence: count) table once, then run BPE over that compact table — counts make per-iteration recounting cheap and bounded by vocab, not corpus size. Use incremental pair-count updates (only adjust counts for pairs touched by the last merge) instead of full recounts. Stream/iterate the file line-by-line; don't hold all tokens in one giant list. For Twitter, decide normalization deliberately (lowercasing? URL/handle handling?) and document it — but keep byte/char fallback so nothing becomes UNK.

**Warning signs:**
Training a single tokenizer takes many minutes/hours locally; memory climbs toward system limit on the 95MB file; vocab dominated by hapax/URL fragments; per-iteration cost scales with corpus size not vocab size.

**Phase to address:**
BPE training-efficiency phase and per-domain preprocessing phase.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Copy a tutorial BPE with whitespace pre-tokenization | Fast working code | Zero bigrams → disqualification; full rewrite | **Never** — opposite of the spec |
| Store compiled `regex.Pattern`/lambda on `self` | Convenient | Pickle won't reload from `code/` → grade 0 | **Never** — use pattern strings + lazy compile |
| Full-corpus pair recount every merge iteration | Simple, obviously correct | O(merges×95MB) — slow/unusable on domain 1 | As the correctness oracle only; not for real training |
| Naive O(n²) rescan encoder | Simplest correct encode | Loses speed axis | Keep as equality oracle for the fast encoder; not for submission |
| Skip byte/char fallback (subword-only vocab) | Smaller vocab, simpler | UNK explosion + alignment break on hidden domain | **Never** for tokenizer 3; risky even for 1/2 |
| Test only `encode(decode)==x`, not cumulative-length monotonicity | Quick round-trip check | Silent NER misalignment, low F1 | **Never** — must test the aligner's actual length logic |
| Tune vocab "as big as possible" for efficiency | Best efficiency number | Cross-word bigrams straddle entity boundaries → F1 tanks | Never blind; only after F1↔efficiency sweep |
| Debug F1/alignment directly on the VM | Real NER numbers | Burns 10h budget; may run out before scoring | Only for final confirmatory runs after local gates pass |

## Integration Gotchas

Mistakes when connecting your tokenizer to the fixed scaffold.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `BaseTokenizer.load` (unpickle from `code/`) | Lambdas / compiled regex / renamed class on `self` | Module-level funcs, pattern strings, stable class path; fresh-process load test |
| `train_ner_model.py` alignment (`decode([id])`, cumulative `decode`) | Decode drops chars / not length-faithful / marker not → space | Length-faithful decode; local test mirroring the cumulative-length aligner |
| `check_submission.py` bigram test (`tok.replace(space_token,' ').strip()`) | Marker not mergeable → no internal space ever appears | Make space marker a mergeable symbol; assert a true bigram exists |
| `check_submission.py` trigram ceiling | Cross-word merges chain to 3+ words | Merge-eligibility filter: reject merges yielding ≥2 internal spaces |
| `generate_tokenizers.py` (unedited regen) | Non-deterministic ties/sampling → different output | Total deterministic tie-break + seeded sampling; double-run diff |
| `train_commands.txt` | Missing/incorrect vocab sizes / `--train_files_3` mix | Encode exact, reproducible commands; regen from them only |
| First-subtoken labeling (-100 on rest) | Cross-word bigram spans an entity boundary | Avoid over-merging across likely entity boundaries |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Full-corpus pair recount per merge | Training minutes→hours | Word-freq table + incremental pair-count updates | Domain 1 (95MB) / many merges |
| O(n²) rescan encode | Encode time super-linear in length | Rank/heap or linked-list merge encoder | Long Twitter lines; timing harness |
| Loading whole 95MB file into one token list | Memory climbs toward limit | Stream line-by-line; collapse to {word:count} | Domain 1 on local CPU |
| Re-running NER on VM to debug | `time_left` drains, no scores | Local gates before any VM run | 3 tokenizers × debug iterations > 10h |

## Security Mistakes

Not a security-sensitive domain (offline coursework). Closest analogues = integrity/correctness hazards:

| Mistake | Risk | Prevention |
|---------|------|------------|
| Editing forbidden graded files | Grade 0 | Keep read-only; diff vs originals before packaging |
| Space marker collides with real input chars | Corrupted decode/alignment | Pick a rare codepoint absent from corpus; assert no collision |
| Adding external data / new libs | Spec violation / non-repro | Use only provided data + numpy/regex/torch/tqdm |
| Hash-randomization-dependent iteration | Run-to-run vocab drift | Explicit deterministic tie-break; seed; sort before iterating |

## UX Pitfalls

"User" = course staff grading and the report reader.

| Pitfall | Impact | Better Approach |
|---------|--------|-----------------|
| Report exceeds 1 page / wrong font/margins | Format penalty (20%) | Arial 10, 1.15 spacing, 2.54cm margins, single column; check length early |
| Missing required report items (top/bottom-5 bigrams, per-model F1+efficiency, hidden-domain analysis, model-2 discussion) | Lost report points | Build report checklist from spec; fill as evidence is produced |
| Reported numbers not reproducible | Distrust / unverifiable claims | Generate report numbers from the reproducible pipeline only |

## "Looks Done But Isn't" Checklist

- [ ] **Bigram exists:** Vocab has ≥1 token with exactly one internal space (run checker's predicate yourself).
- [ ] **No trigram:** No vocab token has ≥2 internal spaces.
- [ ] **space_token:** Non-None, single rare codepoint, present in vocab, no corpus collision.
- [ ] **Decode length-faithful:** Cumulative `decode(ids[:i])` length is monotonic and `decode(all)`==source char-for-char; aligner spot-check lands labels on correct words.
- [ ] **Pickle reload:** Loads in a FRESH process with cwd `code/` and runs encode/decode (no lambda/compiled-regex/renamed-class).
- [ ] **Determinism:** `generate_tokenizers.py` run twice → byte-identical pickles/merge lists.
- [ ] **No UNK on OOV:** Byte/char fallback covers held-out/emoji/URL/foreign text; zero true UNK.
- [ ] **Fast encode == slow encode:** Optimized encoder output equals the O(n²) oracle.
- [ ] **F1 ≥ 0.4** on dev_1 and dev_2 (the gate), measured via the real pipeline.
- [ ] **Zip/structure:** `check_submission.py` passes on the actual zip; tree matches sample; correct ID in every filename; `train_commands.txt` at root; forbidden files byte-identical to originals.
- [ ] **VM budget:** Confirmatory NER runs only; `time_left` checked; buffer reserved.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Zero bigram / trigram leakage | HIGH (algorithm rewrite) | Re-architect merge loop around mergeable space marker + eligibility filter; re-run all gates |
| Decode not length-faithful | MEDIUM | Fix marker→space restoration; add cumulative-length test; re-run NER to confirm F1 recovers |
| Pickle won't load | MEDIUM | Remove lambdas/compiled-regex; stabilize class path; re-pickle; fresh-process test |
| Non-determinism | MEDIUM | Add total tie-break + seeds; sort iterations; double-run diff until clean |
| UNK on hidden domain | MEDIUM | Add byte/char base alphabet; retrain tokenizer 3; OOV stress test |
| Over-merging tanks F1 | LOW–MEDIUM | Re-sweep vocab size; pick efficiency/F1 knee; retrain |
| Slow encode | LOW–MEDIUM | Swap in rank/heap encoder; equality-test vs oracle |
| Burned VM budget | HIGH (may be unrecoverable) | Triage: run only the highest-value tokenizer; report partial results honestly |
| Forbidden file edited / bad zip | LOW (if caught) / fatal (if submitted) | Restore originals from scaffold; rebuild zip via script; re-run `check_submission.py` |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. No bigram (DQ) | Core BPE merge design | Checker's bigram predicate passes on own vocab |
| 2. Trigram leakage (DQ) | Core BPE merge design | No vocab token has >1 internal space |
| 3. Decode not length-faithful (DQ/GATE) | Decode/alignment-contract | Cumulative-length monotonic + exact reconstruction test |
| 4. space_token invalid (DQ) | Core BPE design / local harness | Non-None, in vocab, no collision (unit test) |
| 5. Pickle won't load (DQ/REPRO) | Serialization | Fresh-process `BaseTokenizer.load` + encode/decode |
| 6. Non-determinism (REPRO) | Determinism/reproducibility | `generate_tokenizers.py` double-run byte-diff |
| 7. Over-merging tanks F1 (COMP/GATE) | Per-domain tuning/eval | F1↔efficiency sweep; F1 ≥ baseline |
| 8. UNK on hidden domain (GATE/COMP) | Tokenizer-3 generalization | Zero UNK on OOV stress set; stable alignment |
| 9. O(n²) encode (COMP) | Encode-performance | Sub-quadratic benchmark; output == slow oracle |
| 10. VM budget burn (REPRO/COMP) | Process discipline (cross-cutting) | "Local gates pass" precondition before each VM run; `time_left` log |
| 11. Forbidden edits / bad zip (DQ/REPRO) | Packaging/submission | `check_submission.py` passes; forbidden files byte-identical |
| 12. Twitter artifacts / OOM on 95MB (COMP) | Training-efficiency + preprocessing | Word-freq table + streaming; bounded training time/memory |

## Sources

- `.planning/PROJECT.md` — spec mechanisms: bigram check predicate (`tok.replace(space_token,' ').strip()`), trigram ceiling, NER alignment via `decode([id])` + cumulative `decode`, first-subtoken labeling (-100), reproducibility via `generate_tokenizers.py`/`train_commands.txt`, 10h VM budget, data sizes (95MB/29MB), forbidden files. [HIGH — primary spec source]
- [Token Classification With Subword Tokenizers for Bulgarian — A. Montgomerie](https://amontgomerie.github.io/2020/09/02/token-classification-bg.html) — first-subword labeling with -100 ignore index; alignment of labels across split subwords. [HIGH]
- [Tokenization and Morphological Fidelity in Uralic NLP (arXiv)](https://arxiv.org/pdf/2602.04241) — greedy char-level alignment after removing SentencePiece `▁` (U+2581) / `</w>` markers; first-subword tagging information-loss tradeoff. [MEDIUM]
- [What do BPE Tokenizers Reveal about their Training Data? (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/10e6dfea9a673bef4a7b1cb9234891bc-Paper-Conference.pdf) — tie-breaking in BPE merges; ties become rarer with more data (determinism-sensitivity of tie-breaks). [MEDIUM]
- [Train It and Forget It: Merge Lists ... BPE Inference (arXiv 2508.06621)](https://arxiv.org/html/2508.06621v1) — merge-list-order determinism; a vocab alone specifies a family of encoders, merge order fixes the deterministic one. [MEDIUM]
- [Byte Pair Encoding: Complete Guide — M. Brenndoerfer](https://mbrenndoerfer.com/writing/byte-pair-encoding-subword-tokenization-guide) — BPE training/inference determinism; same merge rules in same order → same tokens. [MEDIUM]

---
*Pitfalls research for: from-scratch BPE tokenizer feeding a fixed binary NER model (NLP HW2)*
*Researched: 2026-06-24*
