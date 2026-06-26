# Phase 3 Context: Per-Domain Tuning

Phase 3 tunes the two domain-specific tokenizers without changing the locked
submission/training entrypoints. The split is deliberate:

- CPU-local: generate candidate tokenizer pickles, measure tokenization
  efficiency and speed on dev text, and produce a deterministic sweep table.
- VM-only: train/evaluate the BiLSTM NER model for the same candidate pickles on
  the Tesla M60 and record F1.
- Selection: choose tok1 and tok2 vocabulary sizes from the F1/efficiency knee,
  then stage the already-measured candidate pickles as the shipped tokenizer
  files.

## Decisions

1. Candidate grid is fixed at `4000`, `8000`, and `16000` vocabulary entries for
   each of tok1 and tok2, for six candidate NER runs total.
2. Candidate pickles are produced by the unedited `generate_tokenizers.py`, not
   by the tuning harness. This preserves SERDE-03/F1 fidelity because
   `generate_tokenizers.py` accepts one `--vocab_size` for all three tokenizers;
   a candidate with per-domain vocab `V` must therefore come from a full
   generator run at `V`.
3. For each `V`, Phase 3 runs:
   `py -m uv run python generate_tokenizers.py --vocab_size V --seed 42 --output_dir tuning/_gen/V`
   and copies only:
   - `tuning/_gen/V/tokenizer_1.pkl` to
     `tuning/tokenizers/tokenizer_1_vocabV.pkl`
   - `tuning/_gen/V/tokenizer_2.pkl` to
     `tuning/tokenizers/tokenizer_2_vocabV.pkl`
   The generated `tokenizer_3.pkl` is discarded during tuning.
4. `code/tune_tokenizers.py` is a measure-only harness. It loads candidate
   pickles, measures tokens/char and tokens/sec on the matching dev text, checks
   decode reconstruction, and writes `tuning/results/cpu_sweep.csv`.
5. VM F1 results feed selection through a deterministic knee rule: keep
   candidates within `0.02` F1 of the best candidate for that domain, then choose
   lower tokens/char, then higher tokens/sec.
6. Final staging in 03-03 copies the F1-measured candidate pickles into
   `trained_tokenizers/tokenizer_1.pkl` and `trained_tokenizers/tokenizer_2.pkl`.
   Each shipped tok1/tok2 must compare byte-identical against a fresh
   `generate_tokenizers.py --vocab_size {chosen} --seed 42` run at the chosen
   vocab. Tok3 remains the Phase 2 baseline and is regenerated from vocab 8000
   only if missing, without overwriting staged tok1/tok2.
7. `train_commands.txt` is updated in Phase 3 for tok1/tok2 with the exact
   per-tokenizer generator commands used to reproduce the staged pickles. Tok3
   finalization remains Phase 4, and full freeze remains Phase 5.

## Open For Confirmation

All Phase 3 planning questions from the draft are resolved by Claude/user
review:

- RESOLVED: vocab grid stays `[4000, 8000, 16000]`.
- RESOLVED: the 10h M60 budget is treated as hard; the VM runbook starts with
  `tok1@8000` to measure wall-time before launching the remaining five runs.
- RESOLVED: selection uses the approved knee rule, within `0.02` F1 of best,
  then lower tokens/char, then higher tokens/sec.
- RESOLVED: `train_commands.txt` is updated during Phase 3 for tok1/tok2 only;
  tok3 finalizes in Phase 4 and full freeze happens in Phase 5.
