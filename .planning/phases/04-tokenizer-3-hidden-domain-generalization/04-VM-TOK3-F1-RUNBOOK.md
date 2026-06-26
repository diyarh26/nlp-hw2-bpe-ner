# Phase 4 Tok3 VM F1 Runbook (visible-domain regression evidence)

Run this only on the Tesla M60 VM. Do not run these NER jobs locally.
Purpose: measure each tok3 candidate's NER F1 on BOTH visible domains (a
regression/robustness signal — tok3 actually serves the hidden domain) and
produce reusable rows for Phase 5. 6 runs = 3 vocab x 2 visible domains.

## Required files (already in the repo after `git pull`)

- `code/train_ner_model.py`
- `tuning/tokenizers/tokenizer_3_vocab4000.pkl`
- `tuning/tokenizers/tokenizer_3_vocab8000.pkl`
- `tuning/tokenizers/tokenizer_3_vocab16000.pkl`
- `data/ner_data/{train,dev}_1_binary.tagged`, `data/ner_data/{train,dev}_2_binary.tagged`

Create dirs:

```bash
mkdir -p tuning/logs/vm_f1_tok3 tuning/results
```

## Preflight (record before any NER run)

```bash
time_left || true            # if the course VM exposes it; record the value
nvidia-smi
uv run python -c "import torch; print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"
```

## Budget + stop-rule

Phase 3's 6 runs took ~13 min total, so this sweep should be similar. After the
first timed run below, STOP and ask before continuing if any of these hold:
- CUDA is False / device is not the Tesla M60 (logs would show `Using device: cpu`);
- projected six-run wall time (`first_run_minutes x 6`) exceeds 25% of remaining VM budget.

## First timed run (tok3 vocab 8000, domain 1) — record wall-time

```bash
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_3_vocab8000.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged 2>&1 | tee tuning/logs/vm_f1_tok3/tokenizer_3_vocab8000_domain1.log'
```

## Remaining 5 runs (only if the stop-rule permits)

```bash
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_3_vocab8000.pkl  --train_file data/ner_data/train_2_binary.tagged --dev_file data/ner_data/dev_2_binary.tagged 2>&1 | tee tuning/logs/vm_f1_tok3/tokenizer_3_vocab8000_domain2.log'
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_3_vocab4000.pkl  --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged 2>&1 | tee tuning/logs/vm_f1_tok3/tokenizer_3_vocab4000_domain1.log'
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_3_vocab4000.pkl  --train_file data/ner_data/train_2_binary.tagged --dev_file data/ner_data/dev_2_binary.tagged 2>&1 | tee tuning/logs/vm_f1_tok3/tokenizer_3_vocab4000_domain2.log'
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_3_vocab16000.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged 2>&1 | tee tuning/logs/vm_f1_tok3/tokenizer_3_vocab16000_domain1.log'
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_3_vocab16000.pkl --train_file data/ner_data/train_2_binary.tagged --dev_file data/ner_data/dev_2_binary.tagged 2>&1 | tee tuning/logs/vm_f1_tok3/tokenizer_3_vocab16000_domain2.log'
```

## Record results

Create `tuning/results/tok3_vm_f1_results.csv` with header:

```csv
domain,tokenizer_index,vocab_size,tokenizer_path,train_file,dev_file,f1,wall_time_minutes,log_path,notes
```

One row per completed run (`tokenizer_index=3`): the printed `Best F1`, measured
wall-time, and log path. Six rows expected (vocab in {4000,8000,16000} x domain
in {1,2}). If the stop-rule fires, write a single row whose `notes` explain it
instead of fabricating missing F1 values. Commit + push, then hand the baton back
to Claude for 04-04 selection.
