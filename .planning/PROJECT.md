# BPE Tokenizer for Cross-Domain NER (NLP HW2)

## What This Is

A from-scratch Byte Pair Encoding (BPE) tokenizer, implemented in `code/bpe_tokenizer.py` as a subclass of the course-provided `BaseTokenizer`, that feeds a **fixed** binary NER model (entity vs. non-entity, word-level F1). We deliver **three** trained tokenizers — one tuned for a Twitter/social domain, one for a news/newswire domain, and one designed to generalize to a **hidden** third domain — and a ≤1-page report. This is a graded university NLP homework (HW2); the deliverable is a single `HW2_<id>.zip` submission.

## Core Value

Three correct, reproducible BPE tokenizers that (a) pass every hard format/spec gate so the submission is not zeroed or disqualified, and (b) score well on the competition's three equally-weighted axes — **tokenizer efficiency (tokens/sentence), tokenization speed, and NER F1** — including on the hidden domain.

## Business Context

<!-- Graded coursework, not a product. -->

- **Customer**: The student (submitter) — graded by course staff.
- **Success metric**: Final homework grade. Weighting: 20% correctness + F1≥0.4 gate, 45% competition (efficiency/speed/F1), 15% tokenizer-3 hidden-domain analysis, 20% report.

## Requirements

### Validated

(None yet — ship to validate)

### Active

<!-- Hypotheses until the submission passes check_submission.py and scores well. -->

**Implementation (correctness gates — 20%)**
- [ ] BPE implemented from characters upward in `code/bpe_tokenizer.py` (subclass `BaseTokenizer`), implementing `train`, `encode`, `decode`.
- [ ] Tokenizer produces **at least one bigram token** (a single token spanning two whitespace-separated words; exactly one internal space) — checked by `check_submission.py`; absence = disqualification.
- [ ] Tokens are capped at the **bigram** level — never 3+ words / never 2+ internal spaces.
- [ ] `space_token` attribute is set (non-`None`) and present in the vocab.
- [ ] `decode` is length-faithful and restores real spaces, so the NER pipeline's token→word char-span alignment works.
- [ ] Tokenizer pickles/unpickles from `code/` alone (no lambdas / unpicklable state); loadable via `BaseTokenizer.load`.
- [ ] F1 ≥ 0.4 on `dev_1_binary.tagged` and `dev_2_binary.tagged`.

**Competition (45%)**
- [ ] Fast `encode` (avoid naive O(n²) rescan) to win the speed axis.
- [ ] High token efficiency (fewer tokens/sentence) via cross-word bigram merges and tuned vocab size, without tanking F1.
- [ ] Tokenizer 1 tuned for Twitter domain; Tokenizer 2 for news domain.
- [ ] Tokenizer 3 designed to generalize to the hidden domain (data-mix strategy, generalizing vocab, byte/char fallback to avoid UNK).

**Reproducibility & packaging**
- [ ] All 3 tokenizers regenerable via the unedited `generate_tokenizers.py`.
- [ ] `train_commands.txt` at zip root with exact training commands (vocab sizes, `--train_files_3`, etc.).
- [ ] Submission zip matches the required structure exactly (passes `check_submission.py`).

**Report (20%)**
- [ ] ≤1 A4 page, Arial 10, 1.15 spacing, 2.54cm margins, single column, captioned centered images, in-image text ≥9pt.
- [ ] Author name + ID; per-tokenizer explanation (how it differs from basic BPE, how it handles the hidden domain); top-5 & bottom-5 most/least common bigrams in training; dev F1 + efficiency per model; hidden-domain evaluation; model-2 performance + parameter/characteristic/improvement discussion.

### Out of Scope

- Editing `base_tokenizer.py`, `train_ner_model.py`, `generate_tokenizers.py`, or `check_submission.py` — graded against originals; only file-path args in `train_ner_model.py` may differ per domain.
- Adding external data or new Python libraries — only provided data and the `uv` env (numpy, regex, torch, tqdm) are allowed.
- Changing fixed NER hyperparameters (SEED, batch size, LR, epochs) — the model is uniform for all students.
- Reading/processing the NER `.tagged` files ourselves — handled by the provided pipeline.

## Context

- **Domains** (from sampling the data):
  - Domain 1 = Twitter/social: `@mentions`, hashtags, URLs, misspellings, emoticons. NER data ≈ Twitter/WNUT style.
  - Domain 2 = News/newswire: formal articles & headlines. NER data ≈ CoNLL-2003 style.
  - Domain 3 = hidden, distinct from both; tokenizer 3 trains on a chosen mix of domains 1+2.
- **Data sizes**: `domain_1_train.txt` ~95MB, `domain_2_train.txt` ~29MB (one sentence per line), plus matching dev files and NER `.tagged` files. Provided in `data (3).zip`.
- **NER alignment mechanism**: `train_ner_model.py` derives token→word spans by calling `decode([id])` and cumulative `decode(token_ids[:i+1])` and comparing lengths to the original text; only the first subtoken of each word carries the label (first-subtoken labeling), rest = -100.
- **Bigram check detail**: `check_submission.py` flags a token as a bigram when `tok.replace(space_token,' ').strip()` still contains an internal space — so the space marker must be a mergeable symbol and merges must be allowed to cross it.
- **Provided scaffold** lives in `code_env_example.zip`: `code/` (base_tokenizer, bpe_tokenizer stub, train_tokenizer, test_tokenizer, train_ner_model), root `generate_tokenizers.py`, `check_submission.py`, `init.sh`, `pyproject.toml`, `uv.lock`, `README.md`, and a sample `HW2_123456789.zip` showing the exact submission layout.
- **Author/ID**: Diyar Husayyan, ID 214034621. Solo submission → `HW2_214034621.zip`, `report_214034621.pdf`.

## Constraints

- **Tech stack**: Python 3.11–3.12, `uv` environment; libs limited to numpy, regex, torch, tqdm (no additions). Pure-Python/numpy BPE.
- **Compute**: Develop & unit-test the tokenizer locally on Windows (CPU). Run NER training / F1 / timing on the **course VM** (Tesla M60 GPU) — **10-hour total runtime budget**, so debug locally first and only use the VM when GPU is needed (`time_left` shows remaining budget).
- **Reproducibility**: Tokenizers must be byte-reproducible from `code/` + `train_commands.txt` via `generate_tokenizers.py` (seeded).
- **Format**: Any file-structure mismatch = grade 0. Report format rules are strict.
- **Integrity**: AI tools allowed, but code is the student's responsibility; no sharing of prompts/AI snippets or code between students.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Allow cross-word BPE merges (deviate from GPT-2 pre-tokenization) | GPT-2-style never merges across spaces → zero bigrams → disqualification; cross-word merges also cut token count (efficiency) | — Pending |
| SentencePiece-style space marker (e.g. `▁`) with decode restoring spaces | NER alignment needs length-faithful decode that maps markers back to real spaces | — Pending |
| Cap merges at one internal space (≤ bigram) | Spec forbids tokens spanning 3+ words | — Pending |
| Compete-hard strategy: multiple variants + tuned vocab + fast encode | Competition is 45% of grade with efficiency/speed/F1 trade-off | — Pending |
| Tokenizer 3 optimizes cross-domain generalization (data mix + generalizing vocab + byte/char fallback) | Hidden domain differs from both training domains | — Pending |
| Build locally, evaluate NER on course VM | Tokenizer training is CPU-only; F1 needs GPU; VM has a 10h budget | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-24 after initialization*
