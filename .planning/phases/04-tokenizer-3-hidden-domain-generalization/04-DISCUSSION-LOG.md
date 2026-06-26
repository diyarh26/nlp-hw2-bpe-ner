# Phase 4: Tokenizer 3 Hidden-Domain Generalization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-06-26
**Phase:** 4-tokenizer-3-hidden-domain-generalization
**Areas discussed:** Training mix strategy, Dedup strategy, Tok3 vocab size, Tok3 selection rule, Evidence package, VM eval scope

---

## Training Mix Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Balanced downsample | Use all smaller domain 2 and a deterministic sampled subset of larger domain 1, so tok3 does not become Twitter-heavy. | yes |
| Full combined corpus | Use all domain 1 + all domain 2; simplest and maximizes data, but domain 1 dominates. | |
| Domain-weighted duplicate lines | Repeat/weight domain 2 lines to balance counts while preserving all domain 1 data. | |

**User's choice:** Balanced downsample.
**Notes:** Tok3 should generalize rather than overfit the larger Twitter/social corpus.

---

## Dedup Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Exact-line dedup only | Remove duplicate identical lines after mixing; simple, deterministic, and low risk. | yes |
| Light normalized dedup | Dedup after lowercasing/whitespace normalization; catches more repeats but may erase domain-specific casing signals. | |
| No dedup | Preserve raw frequency exactly after sampling. | |

**User's choice:** Exact-line dedup only.
**Notes:** Preserve casing and punctuation as possible NER/domain signals.

---

## Tok3 Vocab Size

| Option | Description | Selected |
|--------|-------------|----------|
| 8000 | Moderate, already proven checker-clean, good speed/efficiency balance, lower overfit risk for hidden domain. | |
| 16000 | More efficient tokens/char, but higher risk of memorizing domain-specific phrases and hurting hidden transfer. | |
| Sweep 4000/8000/16000 | More evidence, more local generation time; useful for a tok3-specific knee. | yes |

**User's choice:** Sweep 4000/8000/16000.
**Notes:** Phase 4 should not assume the Phase 2 vocab-8000 tok3 baseline is optimal.

---

## Tok3 Selection Rule

| Option | Description | Selected |
|--------|-------------|----------|
| Hidden-generalization score | Choose the best balance of held-out cross-domain efficiency, zero-UNK/OOV robustness, and no bigram/alignment regressions; avoid choosing only by tokens/char. | yes |
| Efficiency-first within gates | Choose the lowest tokens/char candidate as long as checker/OOV/alignment gates pass. | |
| Moderate default unless evidence strongly disagrees | Prefer 8000, switch only if 4000 or 16000 clearly wins on multiple signals. | |

**User's choice:** Hidden-generalization score.
**Notes:** The chosen tok3 should be justified as robust for a hidden domain, not merely efficient on visible text.

---

## Evidence Package

| Option | Description | Selected |
|--------|-------------|----------|
| Local evidence only | CPU sweep, held-out cross-domain slices, OOV stress strings, reconstruction/alignment/checker gates. Save VM/GPU for Phase 5. | |
| One VM smoke run | Add a single NER run with tok3 on available dev domain as a sanity check, spending a little GPU budget now. | |
| Full tok3 VM evaluation now | Measure tok3 F1 on available domains before Phase 5, more confidence but spends VM budget early. | yes |

**User's choice:** Full tok3 VM evaluation now.
**Notes:** This pulls some VM work forward from Phase 5, so the plan needs a budget stop-rule.

---

## VM Eval Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Tok3 on both visible NER domains | Run tok3 against domain 1 and domain 2 NER dev sets, record F1/wall-time/logs, then Phase 5 can reuse these numbers. | |
| Tok3 plus compare to staged tok1/tok2 | Rerun tok1/tok2 too for same-session comparison; more confidence, more GPU budget. | |
| Tok3 candidates before final selection | Run VM F1 for all tok3 vocab candidates before choosing; strongest signal, most budget. | yes |

**User's choice:** Tok3 candidates before final selection.
**Notes:** Phase 4 should evaluate all tok3 vocab candidates on the VM before final staging.

---

## the agent's Discretion

- Exact deterministic sampling method for the balanced domain mix.
- Exact hidden-generalization score formula.
- Whether to create helper scripts for mix generation and tok3 candidate evaluation.

## Deferred Ideas

- Full all-tokenizer final competition measurement remains Phase 5.
- Final report writing remains Phase 6.
