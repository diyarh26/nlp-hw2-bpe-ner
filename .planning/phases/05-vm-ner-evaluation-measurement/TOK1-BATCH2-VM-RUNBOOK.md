# tok1 Batch 2 — structured pre-tokenization (VM F1 runbook)

**Goal:** test whether the **structured (GPT-2-style) pre-tokenization** lifts tok1
Twitter F1 above the Batch 1.5 plateau (mix vocab tops out at **0.4780 @ 6000**).
Candidates are the balanced mix trained with the new pre-tok (`bpe_tokenizer.py`
`PRETOKEN_PATTERN`; splits letters/digits/punct, binds leading space to ▁) at a
vocab grid. Local gates GREEN (align_check, speed_check --only all: oracle +
train==full-recount + cache). Decode/encode mechanics unchanged → length-faithful.

**Candidates** (in `tuning/tok1_b2/`, scored on the TWITTER domain `train_1/dev_1`):
`mix_v6000`, `mix_v8000`, `mix_v10000`, `mix_v12000` (all structured pre-tok).

**Baselines to beat:** Batch 1.5 best = `\X` mix @ 6000 = **0.4780**; staged tok1 = 0.4352.

**Budget:** 4 short domain-1 NER runs (~1 min each, ~4–5 min). Trivial.

---

## Paste-ready VM prompt

```text
You are on the Tesla M60 VM for the NLP HW2 BPE-tokenizer project. Task: tok1
Batch 2 — structured-pre-tokenization mix candidates on the Twitter domain. Do
exactly this.

SETUP
1. cd ~/hw2 && git pull   (brings new candidates under tuning/tok1_b2/)
2. uv sync if needed. CUDA check: uv run python -c "import torch;print(torch.cuda.is_available())". If False, STOP and report.
3. mkdir -p tuning/logs/tok1_b2 tuning/results

RUN — each variant is a tok1 candidate (balanced mix, structured pre-tok, that
vocab) evaluated on the TWITTER domain (train_1/dev_1). Run all 4:
for v in mix_v6000 mix_v8000 mix_v10000 mix_v12000; do
  /usr/bin/time -p sh -c "PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tok1_b2/$v.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged 2>&1 | tee tuning/logs/tok1_b2/$v.log"
done

RECORD
4. Write tuning/results/tok1_b2_f1.csv with header:
   variant,vocab_size,train_data,pretok,f1,wall_time_minutes,log_path,notes
   One row per variant (train_data=balanced-mix; pretok=structured; vocab = number
   in the name). Use the printed "Best F1" and measured wall-time.
5. Commit (`feat(tok1-opt): Batch 2 structured-pretok F1 sweep results`) and push to
   origin/master. Briefly note the F1s in HANDOFF.md under "What was just done".
   Then you can turn the VM off.

HARD RULES: never edit the 4 forbidden files; don't change NER hyperparameters; if
anything errors, STOP and report.
```

---

## Candidate SHA-256 manifest (integrity)
| variant | vocab | tok/char (d1) | sha256 |
|---------|-------|---------------|--------|
| mix_v6000 | 6000 | 0.3200 | `deb02642b37cc235c8a8200d26c5fb59998879a2d0e7b752b333e37da1475058` |
| mix_v8000 | 8000 | 0.3083 | `052830bb946aa39f3e5205a7ecc6d98f72da5f4331b852978bfbcf6c3d4987b3` |
| mix_v10000 | 10000 | 0.3002 | `5d6d9d10680fabbd03813b9f461067fb8149db3f0daff0921c45923802cac360` |
| mix_v12000 | 12000 | 0.2942 | `28be444732a8ab955fba4ed756ae3430fa1edee2190b05a98e6e0d908a1e5db0` |

(For reference, `\X` mix @ 6000 had tok/char 0.3067 — structured trades a little
efficiency for cleaner word boundaries; the VM F1 decides if it's worth it.)

## When results come back (Claude)
- Compare best structured F1 vs the `\X` Batch 1.5 best (0.4780) and weigh the
  equal-axis efficiency from `tok1_b2_cpu.csv` (structured pre-tok usually trades a
  little efficiency for cleaner boundaries).
- **If a structured tok1 clearly beats 0.4780:** Batch 2 wins for tok1. Because the
  pre-tok is the *shared* default, also generate + VM-score structured **tok2**
  (domain_2 @ 16000) and **tok3** (mix @ 8000) to confirm they don't regress
  (tok2 0.9557, tok3 d1 0.4217 / d2 0.9587). If all hold/improve, stage all 3 via
  the unedited generator (structured default), byte-verify, rebuild zip, gates.
- **If structured does NOT beat 0.4780:** revert the `bpe_tokenizer.py` pre-tok
  change (git revert the Batch 2 commit) and ship `\X` `mix_v6000` (0.4780) — a
  safe, gate-clean +0.043 over the staged tok1. Then resume Phase 5 / Phase 6.
