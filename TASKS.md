# NLP HW2 — BPE Tokenizer for Cross-Domain NER

> **Private repo.** This is graded coursework. Do **not** make it public until after grading
> (academic-integrity rule: no sharing of code between students).

From-scratch **Byte Pair Encoding** tokenizer (`code/bpe_tokenizer.py`, `BPETokenizer(BaseTokenizer)`)
feeding a **fixed** binary NER model (word-level F1). Deliverable: **3 trained tokenizers + a ≤1-page
report**, packaged as `HW2_214034621.zip`. Scored on 3 equal axes — efficiency (tokens/sentence),
tokenization speed, and NER F1 — including a **hidden** third domain.

This repo is also the **Claude ↔ Codex relay**: `HANDOFF.md` is the single source of truth for state.
Read it first.

## Status

- **Phase 3 of 6 — Per-Domain Tuning (tok1 = Twitter/social, tok2 = news).**
- CPU sweep **done**: 6 candidates (tok1/tok2 × vocab 4000/8000/16000) produced by unedited
  `generate_tokenizers.py`, metrics in [`tuning/results/cpu_sweep.csv`](tuning/results/cpu_sweep.csv).
- **Next: VM F1 sweep** — the only step that needs the Tesla M60 GPU.

## ▶ The VM task (run on the Tesla M60 VM only)

Full instructions: [`.planning/phases/03-per-domain-tuning/03-VM-F1-RUNBOOK.md`](.planning/phases/03-per-domain-tuning/03-VM-F1-RUNBOOK.md)

1. `git pull` this repo on the VM, then `py -m uv sync` and confirm `torch.cuda.is_available()`.
2. **Time one run first:** `tokenizer_1_vocab8000` on `train_1/dev_1_binary.tagged`. Record wall-time.
3. **Stop rule:** if `first_run_minutes × 6` would blow the budget margin (keep ≥~5 h of the 10 h total
   for later phases), stop and report before running the other five.
4. Run the remaining 5 commands (tok1 @4000/16000, tok2 @4000/8000/16000) against the matching
   `data/ner_data/*_binary.tagged` files.
5. Record each `Best F1` + wall-time into `tuning/results/vm_f1_results.csv`
   (`domain,tokenizer_index,vocab_size,tokenizer_path,train_file,dev_file,f1,wall_time_minutes,log_path,notes`),
   commit, and push back. Then the relay returns to Claude for selection (`03-03`).

The 6 candidate pkls live in `tuning/tokenizers/`; the tagged NER data is in `data/ner_data/`
(both vendored into this repo for transfer).

## Hard rules (do not violate)

- **Never edit** the four forbidden files (must stay byte-identical):
  `code/base_tokenizer.py`, `code/train_ner_model.py`, `generate_tokenizers.py`, `check_submission.py`.
- Locked stack only: `numpy`, `regex` (`import regex`), `torch`, `tqdm`. No new deps.
- GPU only on the VM. Protect the **10-hour** total M60 budget.
- Don't run the NER F1 sweep locally — it's a VM-only step.

## Map

| What | Where |
|------|-------|
| Relay state (read first) | `HANDOFF.md` |
| Rules / spec | `.claude/CLAUDE.md`, `.planning/PROJECT.md` |
| Tokenizer | `code/bpe_tokenizer.py` |
| Local gates | `code/speed_check.py`, `code/align_check.py` |
| Phase 3 plans | `.planning/phases/03-per-domain-tuning/` |
| VM runbook | `.planning/phases/03-per-domain-tuning/03-VM-F1-RUNBOOK.md` |
