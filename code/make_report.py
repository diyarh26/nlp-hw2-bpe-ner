"""Generate the one-page HW2 report PDF from final measured artifacts."""

from __future__ import annotations

import csv
import html
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDENT_ID = "214034621"
AUTHOR = "Diyar Husayyan"
REPORT_DIR = ROOT / "tuning" / "report"
HTML_PATH = REPORT_DIR / f"report_{STUDENT_ID}.html"
PDF_PATH = ROOT / f"report_{STUDENT_ID}.pdf"
EDGE_PATHS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]


def load_base_tokenizer():
    sys.path.insert(0, str((ROOT / "code").resolve()))
    from base_tokenizer import BaseTokenizer

    return BaseTokenizer


def merge_pair_text(pair: tuple[str, str], space_token: str) -> str:
    parts = [part.replace(space_token, "S") for part in pair]
    return "+".join(parts)


def merge_summary(tokenizer_index: int) -> tuple[str, str]:
    base_tokenizer = load_base_tokenizer()
    tok = base_tokenizer.load(str(ROOT / "trained_tokenizers" / f"tokenizer_{tokenizer_index}.pkl"))
    top = [merge_pair_text(pair, tok.space_token) for pair in tok.merges[:5]]
    bottom = [merge_pair_text(pair, tok.space_token) for pair in tok.merges[-5:]]
    return "; ".join(top), "; ".join(bottom)


def final_rows() -> list[dict[str, str]]:
    with (ROOT / "tuning" / "results" / "final_eval.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def hidden_oov_line() -> str:
    with (ROOT / "tuning" / "results" / "tok3_cpu_sweep.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if "vocab8000" in row["tokenizer_path"]]
    passed = max(int(row["oov_stress_cases_passed"]) for row in selected)
    total = max(int(row["oov_stress_cases_total"]) for row in selected)
    unk = max(int(row["unk_count"]) for row in selected)
    return f"OOV stress {passed}/{total}, UNK={unk}"


def fmt_float(value: str, places: int = 3) -> str:
    return f"{float(value):.{places}f}"


def metric_table(rows: list[dict[str, str]]) -> str:
    lines = []
    for row in rows:
        label = f"tok{row['tokenizer_index']} {row['eval_split'].replace('_dev', '')}"
        lines.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(row['vocab_size'])}</td>"
            f"<td>{html.escape(row['vm_f1'])}</td>"
            f"<td>{fmt_float(row['tokens_per_char'], 3)}</td>"
            f"<td>{fmt_float(row['tokens_per_sentence'], 2)}</td>"
            f"<td>{fmt_float(row['tokens_per_second_cold'], 0)}</td>"
            "</tr>"
        )
    return "\n".join(lines)


def bigram_table() -> str:
    lines = []
    for index in (1, 2, 3):
        top, bottom = merge_summary(index)
        lines.append(
            "<tr>"
            f"<td>tok{index}</td>"
            f"<td>{html.escape(top)}</td>"
            f"<td>{html.escape(bottom)}</td>"
            "</tr>"
        )
    return "\n".join(lines)


def build_html() -> str:
    rows = final_rows()
    oov = hidden_oov_line()
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>HW2 Report {STUDENT_ID}</title>
<style>
@page {{ size: A4; margin: 2.54cm; }}
* {{ box-sizing: border-box; }}
body {{
  font-family: Arial, Helvetica, sans-serif;
  font-size: 10pt;
  line-height: 1.15;
  color: #111;
  margin: 0;
}}
h1 {{ font-size: 12pt; line-height: 1.15; margin: 0 0 4pt 0; }}
p {{ margin: 0 0 5pt 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 3pt 0 6pt 0; table-layout: fixed; }}
th, td {{ border: 0.5pt solid #777; padding: 2.4pt 3pt; vertical-align: top; }}
th {{ background: #eee; font-weight: 700; }}
.metrics th, .metrics td {{ text-align: right; }}
.metrics th:first-child, .metrics td:first-child, .metrics th:nth-child(2), .metrics td:nth-child(2) {{ text-align: left; }}
.bigrams th:nth-child(1), .bigrams td:nth-child(1) {{ width: 9%; }}
.bigrams th:nth-child(2), .bigrams td:nth-child(2) {{ width: 35%; }}
.bigrams th:nth-child(3), .bigrams td:nth-child(3) {{ width: 56%; }}
.note {{ font-size: 9.5pt; }}
</style>
</head>
<body>
<h1>HW2 BPE tokenizer report - {AUTHOR}, {STUDENT_ID}</h1>
<p><b>Implementation.</b> All three tokenizers use from-scratch character-up BPE with a mergeable space marker <b>S</b>, one-cross-space bigram cap, byte fallback that never emits UNK, pickle-safe state, and length-faithful decode for the fixed NER aligner. This differs from basic/pre-tokenized BPE by allowing selected cross-word merges while forbidding 3+ word tokens.</p>
<p><b>Tokenizer choices.</b> tok1 targets Twitter: the balanced domain-1/domain-2 mix at vocab 6000 gave F1 0.4780, +0.043 over the earlier domain-1-only vocab-8000 tokenizer; larger mix vocabs fell sharply, and the structured pre-tokenization experiment was reverted after lowering F1. tok2 targets news with vocab 16000, keeping F1 0.9557 while giving the best measured efficiency/speed tradeoff among news candidates. tok3 handles the hidden domain with the balanced deduped mix, medium vocab 8000, and byte fallback; visible-domain checks are d1=0.4217, d2=0.9587, {html.escape(oov)}.</p>
<table class="metrics">
<tr><th>model/split</th><th>vocab</th><th>dev F1</th><th>tok/char</th><th>tok/sent</th><th>cold tok/s</th></tr>
{metric_table(rows)}
</table>
<p class="note"><b>Top/bottom training bigrams.</b> Ordered BPE merge pairs from the shipped pickles; early pairs are the highest-count retained merges and late pairs are the lowest retained merges. <b>S</b> denotes the learned space marker.</p>
<table class="bigrams">
<tr><th>tok</th><th>top 5 merge pairs</th><th>bottom 5 merge pairs</th></tr>
{bigram_table()}
</table>
<p><b>Hidden-domain and model-2 discussion.</b> tok3 deliberately trades a little visible F1 for fewer tokens than vocab 4000 and broader mixed-domain coverage, with exact reconstruction and no UNK on stress cases. tok2's high news F1 reflects the cleaner news distribution and larger vocab; its 16000 setting improves compression/speed without materially sacrificing F1 versus smaller news tokenizers.</p>
</body>
</html>
"""


def find_browser() -> Path:
    for path in EDGE_PATHS:
        if path.is_file():
            return path
    raise SystemExit("No Edge/Chrome executable found for PDF generation.")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(build_html(), encoding="utf-8")
    browser = find_browser()
    subprocess.run(
        [
            str(browser),
            "--headless",
            "--disable-gpu",
            f"--print-to-pdf={PDF_PATH}",
            HTML_PATH.as_uri(),
        ],
        check=True,
    )
    print(f"Wrote {PDF_PATH.relative_to(ROOT)}")
    print(f"Source HTML: {HTML_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
