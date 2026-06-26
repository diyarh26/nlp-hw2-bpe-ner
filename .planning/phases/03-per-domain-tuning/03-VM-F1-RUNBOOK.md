# Phase 3 VM F1 Runbook

Run this only on the Tesla M60 VM. Do not run these NER jobs locally.

## Required Files

- `code/train_ner_model.py`
- `tuning/tokenizers/tokenizer_1_vocab4000.pkl`
- `tuning/tokenizers/tokenizer_1_vocab8000.pkl`
- `tuning/tokenizers/tokenizer_1_vocab16000.pkl`
- `tuning/tokenizers/tokenizer_2_vocab4000.pkl`
- `tuning/tokenizers/tokenizer_2_vocab8000.pkl`
- `tuning/tokenizers/tokenizer_2_vocab16000.pkl`
- `data/ner_data/train_1_binary.tagged`
- `data/ner_data/dev_1_binary.tagged`
- `data/ner_data/train_2_binary.tagged`
- `data/ner_data/dev_2_binary.tagged`

Create output directories first:

```bash
mkdir -p tuning/logs/vm_f1 tuning/results
```

Record results in `tuning/results/vm_f1_results.csv` with this header:

```csv
domain,tokenizer_index,vocab_size,tokenizer_path,train_file,dev_file,f1,wall_time_minutes,log_path,notes
```

## Budget And Stop Rule

The Tesla M60 budget is 10 hours total. The first step is a timed single run:
tok1 vocab 8000. Record its wall-time before launching anything else.

Stop before running the remaining five commands if:

- the first run fails for environment, data, tokenizer loading, or CUDA reasons;
- projected time for six runs is greater than 10 hours;
- projected time leaves too little margin for log review and result recording.

Use this estimate after the first run:

```text
projected_total_minutes = first_run_minutes * 6
```

## First Timed Run

Run this command first and record wall-time:

```bash
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_1_vocab8000.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged 2>&1 | tee tuning/logs/vm_f1/tokenizer_1_vocab8000.log'
```

## Remaining Commands

Run these only if the stop-rule permits.

```bash
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_1_vocab4000.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged 2>&1 | tee tuning/logs/vm_f1/tokenizer_1_vocab4000.log'
```

```bash
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_1_vocab16000.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged 2>&1 | tee tuning/logs/vm_f1/tokenizer_1_vocab16000.log'
```

```bash
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_2_vocab4000.pkl --train_file data/ner_data/train_2_binary.tagged --dev_file data/ner_data/dev_2_binary.tagged 2>&1 | tee tuning/logs/vm_f1/tokenizer_2_vocab4000.log'
```

```bash
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_2_vocab8000.pkl --train_file data/ner_data/train_2_binary.tagged --dev_file data/ner_data/dev_2_binary.tagged 2>&1 | tee tuning/logs/vm_f1/tokenizer_2_vocab8000.log'
```

```bash
/usr/bin/time -p sh -c 'PYTHONUNBUFFERED=1 uv run python code/train_ner_model.py --tokenizer_path tuning/tokenizers/tokenizer_2_vocab16000.pkl --train_file data/ner_data/train_2_binary.tagged --dev_file data/ner_data/dev_2_binary.tagged 2>&1 | tee tuning/logs/vm_f1/tokenizer_2_vocab16000.log'
```

## Result Recording

For each completed command, append one row to
`tuning/results/vm_f1_results.csv`. Use the F1 printed by
`code/train_ner_model.py` for the `f1` column, the measured wall-time for
`wall_time_minutes`, and the command log path for `log_path`.

If the first timed run triggers the stop-rule, write a single note row or WIP
note in `HANDOFF.md` instead of fabricating missing F1 rows.
