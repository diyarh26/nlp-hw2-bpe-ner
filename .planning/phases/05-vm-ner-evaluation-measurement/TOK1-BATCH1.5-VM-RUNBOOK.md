# tok1 Batch 1.5 — balanced-mix vocab sweep (VM F1 runbook)

**Goal:** find whether the balanced-mix tok1 recipe reaches ~0.50 Twitter F1 by
sweeping vocab. Batch 1 anchor: `mix_v6000` = **0.4780**. Candidates here are the
SAME mix at more vocab sizes, generated stock-reproducibly (provenance-gated:
re-training mix@6000 byte-matches the VM-scored `mix_v6000.pkl`).

**Candidates** (all in `tuning/tok1_sweep/`, scored on the TWITTER domain):
`mix_v6000` (control re-score — noise estimate vs Batch 1's 0.4780),
`mix_v7000`, `mix_v8000`, `mix_v10000`, `mix_v12000`.

**Budget:** 5 short domain-1 NER runs (~1 min each, ~5–6 min total). Trivial.

## Prior evidence — the curve is already half-mapped (expect a plateau)
`mix_v8000` is **byte-identical** to the staged `tokenizer_3.pkl` (`f03aac21…`),
which the Phase 4 tok3 sweep already scored on the Twitter domain. Combining that
with Batch 1, the mix-vocab × Twitter-F1 curve is:

| mix vocab | Twitter (d1) F1 | source |
|-----------|-----------------|--------|
| 4000 | 0.4450 | tok3 VM sweep (`tok3_vm_f1_results.csv`) |
| **6000** | **0.4780** | Batch 1 (`tok1_exp_f1.csv`) |
| 8000 | 0.4217 | tok3 VM sweep (== `mix_v8000`) |
| 16000 | 0.3792 | tok3 VM sweep |

→ Peaks ≈ 6000, declines above. This sweep is mostly **confirmatory** (control
re-score of 6000 gauges GPU noise; 7000 is the one near-peak point that could
plausibly tie 6000; 8000 should reproduce ≈0.42, cross-validating the byte-identity
F1 reuse). Realistic outcome: plateau < 0.50 → proceed to **Batch 2**
(`TOK1-BATCH2-DESIGN.md`). Note `mix_v6000` already beats the currently-staged tok1
(0.4352) by +0.043 — a safe interim improvement if we ever need to stop.

## Candidate SHA-256 manifest (integrity)
| variant | sha256 |
|---------|--------|
| mix_v6000 | `ef7db8c2382f3c7d7a48828d8ae1da5007ff0c65d746fa3a248fae37d673db14` |
| mix_v7000 | `83ed05e3783ff32f48b9023609a5ef7bda3dc26711c1c288c676b52582333354` |
| mix_v8000 | `f03aac21afaa350e5ac5b09ffe501b7400524dc558c18f7b9fc42aed2254e6d9` (== staged tok3) |
| mix_v10000 | `9ec7e763cbb46af1cce02e3d73a62439a0bf544b367b81dcdc52ddd79322a891` |
| mix_v12000 | `48073f564043c9828b172dcb50f58dd61ba1616c77c006b587e0bbc3fd75e177` |

---

## Paste-ready VM prompt

```text
You are on the Tesla M60 VM for the NLP HW2 BPE-tokenizer project. Task: tok1
Batch 1.5 — a balanced-mix vocab sweep on the Twitter domain. Do exactly this.

SETUP
1. cd ~/hw2 && git pull   (brings new tok1 candidates under tuning/tok1_sweep/)
2. uv sync if needed. CUDA check: uv run python -c "import torch;print(torch.cuda.is_available())". If False, STOP and report.
3. mkdir -p tuning/logs/tok1_sweep tuning/results

RUN — each variant is a tok1 candidate (the balanced mix at that vocab) evaluated
on the TWITTER domain (train_1/dev_1). Run all 5 (mix_v6000 is a control re-score):
for v in mix_v6000 mix_v7000 mix_v8000 mix_v10000 mix_v12000; do
  /usr/bin/time -p sh -c "PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tok1_sweep/$v.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged 2>&1 | tee tuning/logs/tok1_sweep/$v.log"
done

RECORD
4. Write tuning/results/tok1_sweep_f1.csv with header:
   variant,vocab_size,train_data,f1,wall_time_minutes,log_path,notes
   One row per variant (all train_data=balanced-mix; vocab = the number in the name;
   mix_v6000 note="control re-score vs Batch1 0.4780"). Use the printed "Best F1" and
   measured wall-time.
5. Commit (`feat(tok1-opt): Batch 1.5 mix-vocab F1 sweep results`) and push to
   origin/master. Briefly note the F1s in HANDOFF.md under "What was just done".
   Then you can turn the VM off.

HARD RULES: never edit the 4 forbidden files; don't change NER hyperparameters; if
anything errors, STOP and report.
```

---

## When results come back (Claude)
- Plot mix F1 vs vocab (with the 6000 control vs Batch 1 0.4780 = noise gauge).
- **If any candidate ≥ ~0.50** (or clearly best on the equal-weighted axes incl.
  efficiency from `tuning/results/tok1_sweep_cpu.csv`): stage it — reproduce via the
  stock generator (`generate_tokenizers.py --vocab_size V --train_files_1 tok3_mix_d1sample_d2all_seed42.txt`,
  take `tokenizer_1.pkl`), byte-verify vs the candidate, copy to
  `trained_tokenizers/tokenizer_1.pkl`, rebuild zip, all gates green. tok1 done.
- **If it plateaus < ~0.50:** implement Batch 2 (see `TOK1-BATCH2-DESIGN.md`) and
  sweep again.
- Either way, then resume Phase 5 final measurement with the chosen tok1.
