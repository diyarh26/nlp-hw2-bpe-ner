"""Build the Phase-1 fallback submission zip.

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
    ROOT / "train_commands.txt",
]


def pdf_stream(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def ensure_inputs() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        raise SystemExit("Missing required input(s): " + ", ".join(missing))


def write_placeholder_report() -> None:
    REPORT.write_bytes(
        pdf_stream("Placeholder report for Diyar Husayyan, ID 214034621. Final report is Phase 6.")
    )


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    entries = [
        (REPORT, REPORT.name),
        (ROOT / "trained_tokenizers" / "tokenizer_1.pkl", "trained_tokenizers/tokenizer_1.pkl"),
        (ROOT / "trained_tokenizers" / "tokenizer_2.pkl", "trained_tokenizers/tokenizer_2.pkl"),
        (ROOT / "trained_tokenizers" / "tokenizer_3.pkl", "trained_tokenizers/tokenizer_3.pkl"),
        (ROOT / "code" / "bpe_tokenizer.py", "code/bpe_tokenizer.py"),
        (ROOT / "train_commands.txt", "train_commands.txt"),
    ]
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, arcname in entries:
            archive.write(source, arcname)


def main() -> None:
    os.chdir(ROOT)
    ensure_inputs()
    write_placeholder_report()
    build_zip()
    print(f"Built {ZIP_PATH.name}")


if __name__ == "__main__":
    main()
