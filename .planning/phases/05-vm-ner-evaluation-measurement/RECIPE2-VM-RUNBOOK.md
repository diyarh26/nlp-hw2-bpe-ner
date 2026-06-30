# Recipe-2 VM F1 runbook (tok1 + tok3 recipe sweep)

Purpose: score NER F1 for the new tok1/tok3 recipe candidates produced by
`code/tune_recipe_sweep.py`. This is the only step that needs the Tesla M60 GPU.
No tokenizer code, NER hyperparameters, forbidden files, `trained_tokenizers/`,
or `HW2_214034621.zip` are touched here — measurement only.

## What is new vs. the previous sweeps

The data-recipe lever was barely explored before (only pure-d1, the 1:1 balanced
mix, and both-full). This adds two unexplored regions:

- **pure domain-1 at higher vocab** — the pure-d1 F1 curve was still climbing at
  v10000 = 0.4633, and higher vocab also helps efficiency + speed.
- **domain-1-heavy mixes (2:1, 3:1)** — between the 1:1 balanced mix (best so
  far, 0.4780 @ 6000) and pure-d1.

## Baselines to beat (already measured, for reference)

| recipe | vocab | domain-1 F1 | domain-2 F1 |
|--------|-------|-------------|-------------|
| 1:1 balanced mix (shipped tok1/tok3) | 6000 | **0.4780** | **0.9621** |
| pure domain-1 | 6000 | 0.4452 | — |
| pure domain-1 | 10000 | 0.4633 | — |

## Candidates to score

tok1 candidates → **domain 1 only** (`train_1`/`dev_1`):

| # | pkl | train/dev |
|---|-----|-----------|
| 1 | `tuning/recipe_sweep/d1_v8000.pkl`   | domain 1 |
| 2 | `tuning/recipe_sweep/d1_v12000.pkl`  | domain 1 |
| 3 | `tuning/recipe_sweep/d1_v16000.pkl`  | domain 1 |
| 4 | `tuning/recipe_sweep/mix2_v5500.pkl` | domain 1 |
| 5 | `tuning/recipe_sweep/mix2_v6000.pkl` | domain 1 |
| 6 | `tuning/recipe_sweep/mix2_v6500.pkl` | domain 1 |
| 7 | `tuning/recipe_sweep/mix3_v6000.pkl` | domain 1 |

tok3 generalist candidates → **domain 2 as well** (the 4 mixes; their domain-1
scores come from rows 4–7 above):

| # | pkl | train/dev |
|---|-----|-----------|
| 8  | `tuning/recipe_sweep/mix2_v5500.pkl` | domain 2 |
| 9  | `tuning/recipe_sweep/mix2_v6000.pkl` | domain 2 |
| 10 | `tuning/recipe_sweep/mix2_v6500.pkl` | domain 2 |
| 11 | `tuning/recipe_sweep/mix3_v6000.pkl` | domain 2 |

Pure-d1 candidates are tok1-only (no news data → not generalist), so they are not
scored on domain 2.

Cost estimate: domain-1 runs ≈ 1 min each (7), domain-2 runs ≈ 3 min each (4) →
**~20 min** total, well within budget.

## Paste-ready VM prompt

```text
You are on the Tesla M60 VM for the NLP HW2 BPE-tokenizer project. Task: score
NER F1 for a tok1/tok3 recipe sweep. Measurement only — do NOT edit the four
forbidden files (code/base_tokenizer.py, code/train_ner_model.py,
generate_tokenizers.py, check_submission.py), do NOT change NER hyperparameters,
do NOT touch trained_tokenizers/ or HW2_214034621.zip. On ANY error, stop and
report instead of filling in missing rows.

SETUP
1. cd ~/hw2 && git fetch origin && git checkout recipe2-sweep && git pull origin recipe2-sweep
   (this work is on the recipe2-sweep BRANCH, not master)
2. uv sync if needed.
3. Confirm CUDA, STOP if false:
   uv run python -c "import torch; print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"
4. Check budget: run `time_left` and report the remaining minutes.
5. mkdir -p tuning/logs/recipe2_f1 tuning/results

TIME THE FIRST RUN, THEN APPLY THE STOP RULE
Run candidate #1 first and record its wall-time. If first_run_minutes * 11 would
use more than ~25% of the remaining budget, STOP and report before continuing.

RUN — domain 1 (tok1 candidates)
for V in d1_v8000 d1_v12000 d1_v16000 mix2_v5500 mix2_v6000 mix2_v6500 mix3_v6000; do
  /usr/bin/time -p sh -c "PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/recipe_sweep/${V}.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged 2>&1 | tee tuning/logs/recipe2_f1/${V}_domain1.log"
done

RUN — domain 2 (tok3 generalist candidates)
for V in mix2_v5500 mix2_v6000 mix2_v6500 mix3_v6000; do
  /usr/bin/time -p sh -c "PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/recipe_sweep/${V}.pkl --train_file data/ner_data/train_2_binary.tagged --dev_file data/ner_data/dev_2_binary.tagged 2>&1 | tee tuning/logs/recipe2_f1/${V}_domain2.log"
done

RECORD
python - <<'PY'
import csv, re
from pathlib import Path
rows = []
d1 = ["d1_v8000","d1_v12000","d1_v16000","mix2_v5500","mix2_v6000","mix2_v6500","mix3_v6000"]
d2 = ["mix2_v5500","mix2_v6000","mix2_v6500","mix3_v6000"]
for v in d1:
    rows.append((v, "1", f"tuning/recipe_sweep/{v}.pkl", "data/ner_data/train_1_binary.tagged", "data/ner_data/dev_1_binary.tagged", f"tuning/logs/recipe2_f1/{v}_domain1.log"))
for v in d2:
    rows.append((v, "2", f"tuning/recipe_sweep/{v}.pkl", "data/ner_data/train_2_binary.tagged", "data/ner_data/dev_2_binary.tagged", f"tuning/logs/recipe2_f1/{v}_domain2.log"))
out = Path("tuning/results/recipe2_vm_f1_results.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["variant","domain","tokenizer_path","train_file","dev_file","f1","log_path"])
    for variant, domain, pkl, trn, dev, log in rows:
        text = Path(log).read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"Best F1:\s*([0-9.]+)", text)
        if not m:
            raise SystemExit(f"Could not find Best F1 in {log}")
        w.writerow([variant, domain, pkl, trn, dev, f"{float(m[-1]):.4f}", log])
print(out.read_text())
PY

COMMIT + PUSH (to the recipe2-sweep branch, NOT master)
git add tuning/logs/recipe2_f1 tuning/results/recipe2_vm_f1_results.csv
git commit -m "feat(recipe2): VM F1 for tok1/tok3 recipe sweep"
git push origin recipe2-sweep

Then report all 11 observed F1 values back.
```
