---
phase: 4
slug: 04-tokenizer-3-hidden-domain-generalization
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-26
---

# Phase 4 Validation: Tokenizer 3 Hidden-Domain Generalization

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | Framework-free Python harnesses and shell/PowerShell commands; no pytest configuration is required. |
| Environment | `py -m uv run python ...` from the locked project environment. |
| Quick run command | `py -m uv run python code/align_check.py; py -m uv run python code/speed_check.py --only all` |
| Full suite command | `py -m uv run python code/build_submission.py; $env:PYTHONIOENCODING='utf-8'; py -m uv run python check_submission.py HW2_214034621.zip` |
| VM-only command family | `uv run python code/train_ner_model.py ...` on the course VM with CUDA/Tesla M60 preflight. |

## Sampling Rate

Run validation at these sampling points:

- After each task commit that changes helper code, tokenizer candidates, staged tokenizers, or packaging commands: run the quick local command when applicable.
- After Plan 04-01: rerun the mix helper and compare manifest/reproducibility evidence.
- After Plan 04-02: run tok3 CPU sweep plus `align_check.py` and `speed_check.py --only all` before any VM spend.
- During Plan 04-03: capture CUDA/device/time-left preflight, run one measured pilot, then apply the stop-rule before completing the remaining VM F1 rows.
- After Plan 04-04: run the full suite command and confirm tok1/tok2 staged hashes remain unchanged.

This cadence is Nyquist-compliant for Phase 4 because every generated artifact has either an immediate reproducibility check, a local gate before downstream use, or a VM-result shape check before selection.

## Per-Task Verification Map

| Task ID | Requirement | Verification Type | Command / Check |
|---------|-------------|-------------------|-----------------|
| 04-01-01 | GEN-01 | unit/smoke | `py -m uv run python code/make_tok3_mix.py --seed 42 --domain1 data/domain_1_train.txt --domain2 data/domain_2_train.txt --output data/tok3_mix_d1sample_d2all_seed42.txt --manifest tuning/results/tok3_mix_manifest.json` |
| 04-01-02 | GEN-01 | reproducibility | Rerun the helper to `tuning/results/tok3_mix_repro_check.txt` or a temp output and compare counts/hashes to `tuning/results/tok3_mix_manifest.json`. |
| 04-02-01 | GEN-01, GEN-02 | generator smoke | Run `generate_tokenizers.py` for vocab sizes `4000`, `8000`, and `16000` with `--train_files_3 tok3_mix_d1sample_d2all_seed42.txt`; verify each `tuning/tokenizers/tokenizer_3_vocab{V}.pkl` exists. |
| 04-02-02 | GEN-02, GEN-03 | CPU evidence | `py -m uv run python code/tune_tok3.py --vocab_sizes 4000 8000 16000 --candidate_dir tuning/tokenizers --results_csv tuning/results/tok3_cpu_sweep.csv` |
| 04-02-03 | GEN-02 | local gates | `py -m uv run python code/align_check.py; py -m uv run python code/speed_check.py --only all` |
| 04-03-01 | GEN-03 | runbook check | `Test-Path .planning/phases/04-tokenizer-3-hidden-domain-generalization/04-VM-TOK3-F1-RUNBOOK.md` and inspect that CUDA/time-left/stop-rule commands are present. |
| 04-03-02 | GEN-03 | manual VM execution | On the VM, run the six tok3 visible-domain F1 jobs only after the pilot stop-rule passes; store logs under `tuning/logs/vm_f1_tok3/`. |
| 04-03-03 | GEN-03 | result shape | `py -c "import csv, pathlib; rows=list(csv.DictReader(open('tuning/results/tok3_vm_f1_results.csv', newline='', encoding='utf-8'))); assert len(rows)==6; assert sorted((r['vocab_size'], r['domain']) for r in rows)==sorted((str(v), str(d)) for v in (4000,8000,16000) for d in (1,2)); assert all(pathlib.Path(r['log_path']).exists() for r in rows); assert all(r['tokenizer_index']=='3' for r in rows); print('tok3 vm results shape ok')"` |
| 04-04-01 | GEN-03 | scoring evidence | `py -m uv run python code/select_tok3.py --cpu_csv tuning/results/tok3_cpu_sweep.csv --vm_csv tuning/results/tok3_vm_f1_results.csv --output_csv tuning/results/tok3_selection.csv --candidate_dir tuning/tokenizers` |
| 04-04-02 | GEN-01, GEN-02 | fresh reproduction | Regenerate the selected tok3 candidate into `tuning/_tok3_cmp/selected` and byte-compare `tokenizer_3.pkl` to the staged selected candidate. |
| 04-04-03 | GEN-02, GEN-03 | final gates | `py -m uv run python code/align_check.py; py -m uv run python code/speed_check.py --only all; py -m uv run python code/build_submission.py; $env:PYTHONIOENCODING='utf-8'; py -m uv run python check_submission.py HW2_214034621.zip` |

## Wave 0 Requirements

Existing validation infrastructure is sufficient to start Phase 4:

- `code/align_check.py` validates tokenizer alignment, reconstruction, and checker-sensitive token constraints.
- `code/speed_check.py` validates runtime budget behavior.
- `code/build_submission.py` and `check_submission.py` validate final packaging.
- `code/train_ner_model.py` is the fixed VM F1 entrypoint.

New phase-specific harnesses (`code/make_tok3_mix.py`, `code/tune_tok3.py`, and `code/select_tok3.py`) are created inside the planned tasks and each has a same-task smoke check before downstream artifacts depend on it.

## Manual-Only Verifications

| Verification | Why Manual | Evidence Required |
|--------------|------------|-------------------|
| Course VM CUDA/time budget preflight | Local machine has no course GPU and `time_left` is VM-specific state. | Runbook log or `HANDOFF.md` entry recording `time_left` if available, CUDA availability, and GPU name. |
| Six tok3 VM F1 runs | Requires the course Tesla M60 VM and budget stewardship. | `tuning/results/tok3_vm_f1_results.csv` with six rows and logs under `tuning/logs/vm_f1_tok3/`. |
| Stop-rule approval if pilot is slow or CUDA is unavailable | Human budget decision is required before spending more VM time. | Explicit note in `HANDOFF.md` or runbook result section before continuing. |

## Validation Sign-Off

- [x] Every Phase 4 requirement has at least one automated or manual verification.
- [x] Generated data and tokenizer artifacts have reproducibility checks before selection.
- [x] VM work has a pilot stop-rule and post-run CSV/log shape validation.
- [x] Final staging requires local alignment, speed, packaging, and checker gates.
