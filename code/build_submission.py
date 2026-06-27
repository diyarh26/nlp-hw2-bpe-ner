"""Build the submission zip.

This helper is not part of the course scaffold. It packages only the files the
checker expects and deliberately excludes fixed/forbidden scaffold files.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDENT_ID = "214034621"
REPORT = ROOT / f"report_{STUDENT_ID}.pdf"
ZIP_PATH = ROOT / f"HW2_{STUDENT_ID}.zip"

REQUIRED_FILES = [
    ROOT / "trained_tokenizers" / "tokenizer_1.pkl",
    ROOT / "trained_tokenizers" / "tokenizer_2.pkl",
    ROOT / "trained_tokenizers" / "tokenizer_3.pkl",
    ROOT / "code" / "bpe_tokenizer.py",
    ROOT / "code" / "make_tok3_mix.py",
    ROOT / "train_commands.txt",
]


def ensure_inputs() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.is_file()]
    if not REPORT.is_file():
        missing.append(REPORT.name)
    if missing:
        raise SystemExit("Missing required input(s): " + ", ".join(missing))


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    entries = [
        (REPORT, REPORT.name),
        (ROOT / "trained_tokenizers" / "tokenizer_1.pkl", "trained_tokenizers/tokenizer_1.pkl"),
        (ROOT / "trained_tokenizers" / "tokenizer_2.pkl", "trained_tokenizers/tokenizer_2.pkl"),
        (ROOT / "trained_tokenizers" / "tokenizer_3.pkl", "trained_tokenizers/tokenizer_3.pkl"),
        (ROOT / "code" / "bpe_tokenizer.py", "code/bpe_tokenizer.py"),
        (ROOT / "code" / "make_tok3_mix.py", "code/make_tok3_mix.py"),
        (ROOT / "train_commands.txt", "train_commands.txt"),
    ]
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, arcname in entries:
            archive.write(source, arcname)


def main() -> None:
    os.chdir(ROOT)
    ensure_inputs()
    build_zip()
    print(f"Built {ZIP_PATH.name}")


if __name__ == "__main__":
    main()
