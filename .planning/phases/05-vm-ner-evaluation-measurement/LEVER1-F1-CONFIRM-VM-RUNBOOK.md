# Lever 1 VM F1 confirmation runbook

Purpose: re-score the shipped, byte-identical tokenizers after the O(n log n)
encoder change. The encoder proof already implies unchanged NER behavior; this
VM pass is belt-and-suspenders evidence for the report/handoff.

Expected F1 values from the final VM sweeps:

| row | tokenizer | split | expected F1 |
|-----|-----------|-------|-------------|
| 1 | tokenizer_1.pkl | domain 1 | 0.4780 |
| 2 | tokenizer_2.pkl | domain 2 | 0.9557 |
| 3 | tokenizer_3.pkl | domain 1 | 0.4217 |
| 4 | tokenizer_3.pkl | domain 2 | 0.9587 |

## Paste-ready VM prompt

```text
You are on the Tesla M60 VM for the NLP HW2 BPE-tokenizer project. Task:
confirm the shipped Lever-1 tokenizers still score the same NER F1. Lever 1 is
byte-identical at encode output, so these four runs should match the prior
final_eval.csv values exactly or within ordinary deterministic-script output.

SETUP
1. cd ~/hw2 && git pull
2. uv sync if needed.
3. Confirm CUDA, then STOP if CUDA is false:
   uv run python -c "import torch; print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"
4. mkdir -p tuning/logs/lever1_f1_confirm tuning/results

RUN
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path trained_tokenizers/tokenizer_1.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged 2>&1 | tee tuning/logs/lever1_f1_confirm/tok1_domain1.log'
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path trained_tokenizers/tokenizer_2.pkl --train_file data/ner_data/train_2_binary.tagged --dev_file data/ner_data/dev_2_binary.tagged 2>&1 | tee tuning/logs/lever1_f1_confirm/tok2_domain2.log'
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path trained_tokenizers/tokenizer_3.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged 2>&1 | tee tuning/logs/lever1_f1_confirm/tok3_domain1.log'
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path trained_tokenizers/tokenizer_3.pkl --train_file data/ner_data/train_2_binary.tagged --dev_file data/ner_data/dev_2_binary.tagged 2>&1 | tee tuning/logs/lever1_f1_confirm/tok3_domain2.log'

RECORD
python - <<'PY'
import csv
import re
from pathlib import Path

rows = [
    ("1", "trained_tokenizers/tokenizer_1.pkl", "domain_1", "data/ner_data/train_1_binary.tagged", "data/ner_data/dev_1_binary.tagged", "0.4780", "tuning/logs/lever1_f1_confirm/tok1_domain1.log"),
    ("2", "trained_tokenizers/tokenizer_2.pkl", "domain_2", "data/ner_data/train_2_binary.tagged", "data/ner_data/dev_2_binary.tagged", "0.9557", "tuning/logs/lever1_f1_confirm/tok2_domain2.log"),
    ("3", "trained_tokenizers/tokenizer_3.pkl", "domain_1", "data/ner_data/train_1_binary.tagged", "data/ner_data/dev_1_binary.tagged", "0.4217", "tuning/logs/lever1_f1_confirm/tok3_domain1.log"),
    ("3", "trained_tokenizers/tokenizer_3.pkl", "domain_2", "data/ner_data/train_2_binary.tagged", "data/ner_data/dev_2_binary.tagged", "0.9587", "tuning/logs/lever1_f1_confirm/tok3_domain2.log"),
]

out = Path("tuning/results/lever1_f1_confirm.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["tokenizer_index", "tokenizer_path", "eval_split", "train_file", "dev_file", "expected_f1", "observed_f1", "matches_expected_4dp", "log_path"])
    for tokenizer_index, tokenizer_path, split, train_file, dev_file, expected, log_path in rows:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"Best F1:\s*([0-9.]+)", text)
        if not matches:
            raise SystemExit(f"Could not find Best F1 in {log_path}")
        observed = f"{float(matches[-1]):.4f}"
        writer.writerow([tokenizer_index, tokenizer_path, split, train_file, dev_file, expected, observed, str(observed == expected).lower(), log_path])
print(out.read_text())
PY

5. Commit and push:
   git add tuning/logs/lever1_f1_confirm tuning/results/lever1_f1_confirm.csv
   git commit -m "test(vm): confirm Lever 1 NER F1"
   git push origin master

Report the four observed F1 values back to Codex. HARD RULES: never edit the
four forbidden files, never change NER hyperparameters, and stop/report on any
error instead of filling missing rows.
```

