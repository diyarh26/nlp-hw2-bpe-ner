---
phase: 03-per-domain-tuning
plan: 02
subsystem: vm-evaluation
tags: [vm, ner, f1, sweep]
provides:
  - VM F1 runbook
  - Six candidate NER F1 measurements on Tesla M60
requirements-completed:
  - TUNE-03
completed: 2026-06-26
status: complete
---

# Phase 3 Plan 2: VM F1 Sweep Summary

## What was delivered

- Wrote `.planning/phases/03-per-domain-tuning/03-VM-F1-RUNBOOK.md` with the six exact VM commands, log paths, CSV format, and budget stop-rule.
- Ran the sweep on the Tesla M60 VM after confirming CUDA as `True Tesla M60`.
- Timed tok1 vocab 8000 first as required. The first run took `0.9443` minutes, so the projected six-run sweep stayed safely under the 10h budget.
- Wrote `tuning/results/vm_f1_results.csv` and logs under `tuning/logs/vm_f1/`.

## VM F1 Results

| tokenizer | vocab | F1 | wall time minutes |
|-----------|------:|---:|------------------:|
| tok1 | 4000 | 0.4237 | 0.8820 |
| tok1 | 8000 | 0.4352 | 0.9443 |
| tok1 | 16000 | 0.3692 | 0.8217 |
| tok2 | 4000 | 0.9610 | 3.6005 |
| tok2 | 8000 | 0.9570 | 3.3208 |
| tok2 | 16000 | 0.9557 | 3.2012 |

## Verification

- Six VM rows were recorded.
- Every row points to a captured log file.
- No forbidden files were edited.
- No NER hyperparameters were changed.

