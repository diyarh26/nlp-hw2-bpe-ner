# Phase 4: Tokenizer 3 Hidden-Domain Generalization - Context

**Gathered:** 2026-06-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Build and stage `trained_tokenizers/tokenizer_3.pkl` for hidden-domain
generalization. The phase owns tokenizer 3's training mix, candidate sweep,
OOV/held-out stress evidence, VM F1 evidence for tok3 candidates, final tok3
selection, and the notes needed later for the report.

Explicitly in scope:
- create a balanced domain-1/domain-2 training recipe for tok3;
- remove exact duplicate lines from the tok3 training mix;
- train tok3 candidates at vocab sizes `4000`, `8000`, and `16000`;
- evaluate candidates locally for efficiency, reconstruction, OOV robustness,
  and checker/alignment safety;
- run VM F1 evaluation for all tok3 vocab candidates before final selection;
- select tok3 using a hidden-generalization score, then stage the selected
  candidate as `trained_tokenizers/tokenizer_3.pkl`.

Explicitly out of scope:
- changing the fixed course scaffold files;
- changing NER model hyperparameters;
- changing the already staged tok1 vocab 8000 or tok2 vocab 16000;
- final report writing and final all-tokenizer freeze, which remain Phase 6
  and Phase 5 respectively.

</domain>

<decisions>
## Implementation Decisions

### Training Mix Strategy

- **D-01:** Tok3 should train on a balanced domain mix: use all of domain 2
  and a deterministic sampled subset of larger domain 1 so tokenizer 3 does
  not become Twitter-heavy.
- **D-02:** Deduplication should be exact-line dedup only after mixing. Do not
  lowercase or normalize for dedup, because casing and punctuation may carry
  domain-specific NER signal.

### Candidate Sweep and Selection

- **D-03:** Tok3 should sweep vocab sizes `4000`, `8000`, and `16000` rather
  than defaulting to the current Phase 2 vocab-8000 baseline.
- **D-04:** Final tok3 selection should use a hidden-generalization score:
  balance held-out cross-domain efficiency, VM F1, OOV robustness, zero-UNK
  behavior, reconstruction/alignment safety, and checker gates. Do not choose
  solely by lowest tokens/char.

### VM Evidence

- **D-05:** Phase 4 should run VM F1 evaluation for all tok3 vocab candidates
  before final selection. This intentionally pulls some VM work forward from
  Phase 5; the plan must include a budget stop-rule to protect the 10h Tesla
  M60 allocation.
- **D-06:** Tok3 VM F1 should cover visible NER domains for each tok3
  candidate, with deterministic result rows and logs, so Phase 5 can reuse the
  evidence instead of repeating avoidable runs.

### the agent's Discretion

- Define the exact deterministic sampling mechanism for choosing domain-1
  lines, as long as it is reproducible from project files and documented in
  `train_commands.txt` or a helper script.
- Define the concrete hidden-generalization score formula, as long as it
  implements D-04 and remains auditable from CSV evidence.
- Choose whether to create a helper file for mix generation and/or tok3
  evaluation, while preserving the unedited course scaffold files.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning

- `.planning/PROJECT.md` - project constraints, hidden-domain motivation,
  fixed scaffold boundaries, VM budget, and report requirements.
- `.planning/REQUIREMENTS.md` - Phase 4 requirement IDs `GEN-01`, `GEN-02`,
  and `GEN-03`.
- `.planning/ROADMAP.md` - Phase 4 goal and success criteria.
- `.planning/STATE.md` - current position and carry-forward concerns.
- `HANDOFF.md` - relay notes and local/VM caveats.

### Prior phase decisions

- `.planning/phases/01-gates-first-submittable-tokenizer/01-CONTEXT.md` -
  core BPE decisions: char-up BPE, mergeable space marker, one-space cap,
  zero-UNK byte fallback, length-faithful decode, pickle safety.
- `.planning/phases/02-fast-encode-training-efficiency/02-CONTEXT.md` -
  trainer/encode constraints and full-corpus training strategy.
- `.planning/phases/03-per-domain-tuning/03-CONTEXT.md` - generator-backed
  candidate pattern, measure-only harness approach, and staging rules.
- `.planning/phases/03-per-domain-tuning/03-03-SUMMARY.md` - staged tok1/tok2
  choices and current tok3 baseline.

### Fixed scaffold and code

- `generate_tokenizers.py` - unedited course generator. It accepts one
  `--vocab_size` per run and `--train_files_3` for tok3 data files.
- `code/bpe_tokenizer.py` - tokenizer implementation. Preserve byte fallback,
  one-internal-space merge cap, cached encode, deterministic trainer, and
  pickle-safe state.
- `code/tune_tokenizers.py` - Phase 3 measure-only harness pattern to reuse or
  extend for tok3 CPU evidence.
- `code/align_check.py` - local alignment/decode gate that must stay green.
- `code/speed_check.py` - encode/cache/train oracle gate that must stay green
  if tokenizer internals change.
- `check_submission.py` - unedited course checker; final staged tok3 must pass.
- `code/train_ner_model.py` - unedited NER evaluation entrypoint for VM F1.

### Evidence artifacts

- `tuning/results/cpu_sweep.csv` - Phase 3 CPU metric format and examples.
- `tuning/results/vm_f1_results.csv` - Phase 3 VM result format and examples.
- `tuning/logs/vm_f1/` - Phase 3 VM logging convention.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `generate_tokenizers.py` already supports `--train_files_3`, so tok3 can be
  trained on a generated mix file without editing the scaffold.
- `BPETokenizer` already guarantees byte fallback, deterministic training,
  pickle-safe state, and one-internal-space merge legality.
- `code/tune_tokenizers.py` already writes CSV evidence for generated candidate
  pickles and can guide a tok3-specific measure-only harness.
- `code/build_submission.py` packages the currently staged
  `trained_tokenizers/tokenizer_{1,2,3}.pkl` files into the submission zip.

### Established Patterns

- Candidate pickles should come from unedited `generate_tokenizers.py`, then be
  copied into `tuning/tokenizers/` under explicit names.
- Selection/staging should compare staged pickles byte-for-byte with fresh
  generator output for the chosen recipe.
- Local gates (`speed_check.py`, `align_check.py`, `check_submission.py`) must
  be green before spending VM time.
- VM evaluation should write deterministic CSV rows and logs under
  `tuning/results/` and `tuning/logs/`.

### Integration Points

- A mix-generation helper can write a deterministic text file under `tuning/`
  or another documented path, then pass that file name via `--train_files_3`.
- Tok3 candidate pickles should not overwrite staged tok1/tok2.
- `train_commands.txt` must eventually record the exact tok3 reproduction
  command and mix recipe.

</code_context>

<specifics>
## Specific Ideas

- Use all `data/domain_2_train.txt` lines and a deterministic subset of
  `data/domain_1_train.txt` with comparable line count before exact-line dedup.
- Consider writing the mixed training file as a reproducible generated artifact,
  not as a hand-edited corpus.
- For OOV stress, include unseen Unicode, emoji-like characters, URLs,
  mentions, capitalization, punctuation-heavy news snippets, and mixed
  Twitter/news style strings.
- The hidden-generalization score should make it hard for a very large vocab to
  win on efficiency alone if it shows weaker robustness or F1.

</specifics>

<deferred>
## Deferred Ideas

- Full final competition measurement for all three tokenizers remains Phase 5,
  although Phase 4 will produce reusable tok3 VM evidence.
- Final report composition remains Phase 6.
- Advanced tokenizer optimizations such as merge regularization or
  Twitter-specific preprocessing remain v2 unless the Phase 4 evidence shows a
  clear need.

</deferred>

---

*Phase: 4-Tokenizer 3 Hidden-Domain Generalization*
*Context gathered: 2026-06-26*
