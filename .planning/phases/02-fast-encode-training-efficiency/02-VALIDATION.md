---
phase: 02-fast-encode-training-efficiency
created: 2026-06-25
status: planned
---

# Phase 2 Validation Matrix

## VP-2-1: Encode Oracle Equality

Command:

```powershell
py -m uv run python code\speed_check.py --only oracle
```

Pass criteria:

- Fast/public `encode(text)` equals a slow explicit merge-replay oracle for dev samples, NER tagged samples, non-ASCII stress text, and double-space text.
- `decode(encode(text))` keeps the Phase 1 reconstruction contract.

## VP-2-2: Encode Cache Speed

Command:

```powershell
py -m uv run python code\speed_check.py --only cache
```

Pass criteria:

- Repeated cached encode calls are faster than repeated uncached helper calls on a Zipfian/repeated dev sample.
- The cache does not affect output equality.
- Pickled tokenizer state does not contain `_encode_cache`.

## VP-2-3: Training Oracle and Full-Corpus Feasibility

Command:

```powershell
py -m uv run python code\speed_check.py --only train
```

Pass criteria:

- Incremental training matches the old full-recount oracle on a small deterministic fixture.
- `py -m uv run python generate_tokenizers.py --vocab_size 8000 --seed 42 --output_dir trained_tokenizers` completes without relying on `TRAIN_TEXT_LIMIT`.
- All three generated tokenizers load via `BaseTokenizer.load`.

## VP-2-4: Phase 1 Regression Gates

Commands:

```powershell
py -m uv run python code\align_check.py
py -m uv run python code\build_submission.py
$env:PYTHONIOENCODING='utf-8'; py -m uv run python check_submission.py HW2_214034621.zip
py -c "import hashlib; bad=[]; lines=[l.split() for l in open('.planning/phases/01-gates-first-submittable-tokenizer/forbidden_files.sha256') if l.strip()]; [bad.append(p) for h,p in lines if hashlib.sha256(open(p,'rb').read()).hexdigest()!=h]; assert not bad, 'FORBIDDEN FILE MODIFIED: '+str(bad); print('forbidden files byte-identical to baseline')"
```

Pass criteria:

- Alignment harness passes.
- Submission zip rebuilds.
- Checker exits 0 and emits no `[ERROR]`.
- Forbidden-file hashes match.

