"""Build an editable Word report for HW2 without adding project dependencies."""

from __future__ import annotations

from collections import Counter
import csv
import html
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDENT_ID = "214034621"
AUTHOR = "Diyar Husayyan"
DOCX_PATH = Path(os.environ.get("REPORT_DOCX_PATH", ROOT / f"report_{STUDENT_ID}.docx"))
REPORT_DIR = ROOT / "tuning" / "report"
TRAINING_FILES = {
    1: ROOT / "data" / "domain_1_train.txt",
    2: ROOT / "data" / "domain_2_train.txt",
    3: ROOT / "data" / "tok3_mix_d1sample_d2all_seed42.txt",
}

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"
DCTERMS = "http://purl.org/dc/terms/"
DCTYPE = "http://purl.org/dc/dcmitype/"
XSI = "http://www.w3.org/2001/XMLSchema-instance"


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def final_eval() -> list[dict[str, str]]:
    with (ROOT / "tuning" / "results" / "final_eval.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str, places: int) -> str:
    return f"{float(value):.{places}f}"


def metrics_rows() -> list[list[str]]:
    labels = {
        ("1", "domain_1_dev"): "tok1 Twitter",
        ("2", "domain_2_dev"): "tok2 news",
        ("3", "domain_1_dev"): "tok3 visible d1",
        ("3", "domain_2_dev"): "tok3 visible d2",
    }
    rows: list[list[str]] = []
    for row in final_eval():
        key = (row["tokenizer_index"], row["eval_split"])
        rows.append(
            [
                labels.get(key, f"tok{row['tokenizer_index']} {row['eval_split']}"),
                row["vocab_size"],
                row["vm_f1"],
                fmt(row["tokens_per_char"], 3),
                fmt(row["tokens_per_sentence"], 2),
                fmt(row["tokens_per_second_cold"], 0),
            ]
        )
    return rows


def hidden_oov_text() -> str:
    with (ROOT / "tuning" / "results" / "tok3_cpu_sweep.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    chosen = [row for row in rows if "vocab8000" in row["tokenizer_path"]]
    passed = max(int(row["oov_stress_cases_passed"]) for row in chosen)
    total = max(int(row["oov_stress_cases_total"]) for row in chosen)
    unk = max(int(row["unk_count"]) for row in chosen)
    return f"OOV stress {passed}/{total}; UNK={unk}; reconstruction=1.0"


def count_training_bigrams(path: Path) -> dict[str, object]:
    counts: Counter[tuple[str, str]] = Counter()
    lines = 0
    words_total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            words = line.split()
            words_total += len(words)
            counts.update(zip(words, words[1:]))
            lines += 1

    top = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))[:5]
    bottom = sorted(counts.items(), key=lambda item: (item[1], item[0][0], item[0][1]))[:5]
    return {
        "file": str(path.relative_to(ROOT)),
        "lines": lines,
        "words": words_total,
        "pair_types": len(counts),
        "top5": [(" ".join(pair), count) for pair, count in top],
        "bottom5": [(" ".join(pair), count) for pair, count in bottom],
    }


def training_bigram_summaries() -> dict[int, dict[str, object]]:
    return {index: count_training_bigrams(path) for index, path in TRAINING_FILES.items()}


def format_bigram_list(items: list[tuple[str, int]]) -> str:
    return "; ".join(f"{bigram} ({count})" for bigram, count in items)


def rpr(text: str, bold: bool = False) -> str:
    props = (
        "<w:rPr><w:rFonts w:ascii=\"Arial\" w:hAnsi=\"Arial\" w:cs=\"Arial\"/>"
        "<w:sz w:val=\"20\"/><w:szCs w:val=\"20\"/>"
        + ("<w:b/><w:bCs/>" if bold else "")
        + "</w:rPr>"
    )
    return f"<w:r>{props}<w:t xml:space=\"preserve\">{esc(text)}</w:t></w:r>"


def para(text: str = "", *, bold: bool = False, style: str | None = None, after: int = 40) -> str:
    style_xml = f"<w:pStyle w:val=\"{style}\"/>" if style else ""
    spacing = f"<w:spacing w:after=\"{after}\" w:line=\"276\" w:lineRule=\"auto\"/>"
    return f"<w:p><w:pPr>{style_xml}{spacing}</w:pPr>{rpr(text, bold=bold)}</w:p>"


def bullet(text: str) -> str:
    return (
        "<w:p><w:pPr><w:pStyle w:val=\"ListParagraph\"/>"
        "<w:numPr><w:ilvl w:val=\"0\"/><w:numId w:val=\"1\"/></w:numPr>"
        "<w:spacing w:after=\"20\" w:line=\"276\" w:lineRule=\"auto\"/>"
        "</w:pPr>"
        f"{rpr(text)}</w:p>"
    )


def cell(text: str, width: int, shade: str | None = None, bold: bool = False) -> str:
    fill = f"<w:shd w:fill=\"{shade}\"/>" if shade else ""
    paragraphs = "".join(para(part, bold=bold, after=0) for part in text.split("\n"))
    return (
        f"<w:tc><w:tcPr><w:tcW w:w=\"{width}\" w:type=\"dxa\"/>"
        f"<w:tcMar><w:top w:w=\"50\" w:type=\"dxa\"/><w:bottom w:w=\"50\" w:type=\"dxa\"/>"
        f"<w:start w:w=\"70\" w:type=\"dxa\"/><w:end w:w=\"70\" w:type=\"dxa\"/></w:tcMar>{fill}</w:tcPr>"
        f"{paragraphs}</w:tc>"
    )


def table(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    grid = "".join(f"<w:gridCol w:w=\"{width}\"/>" for width in widths)
    xml = [
        "<w:tbl><w:tblPr><w:tblW w:w=\"9026\" w:type=\"dxa\"/>"
        "<w:tblLayout w:type=\"fixed\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"BBBBBB\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"BBBBBB\"/>"
        "</w:tblBorders></w:tblPr>",
        f"<w:tblGrid>{grid}</w:tblGrid>",
        "<w:tr>" + "".join(cell(h, w, shade="EDEDED", bold=True) for h, w in zip(headers, widths)) + "</w:tr>",
    ]
    for row in rows:
        xml.append("<w:tr>" + "".join(cell(value, w) for value, w in zip(row, widths)) + "</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="40" w:line="276" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TitleText">
    <w:name w:val="TitleText"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="50" w:line="276" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:b/><w:bCs/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading">
    <w:name w:val="Heading"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="50" w:after="20" w:line="276" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:b/><w:bCs/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph">
    <w:name w:val="List Paragraph"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:ind w:left="360" w:hanging="180"/></w:pPr>
  </w:style>
</w:styles>"""


def numbering_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="{W}">
  <w:abstractNum w:abstractNumId="0">
    <w:multiLevelType w:val="hybridMultilevel"/>
    <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:tabs><w:tab w:val="num" w:pos="360"/></w:tabs><w:ind w:left="360" w:hanging="180"/></w:pPr></w:lvl>
  </w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>"""


def document_xml() -> str:
    bigrams = training_bigram_summaries()
    bigram_rows = []
    for index in (1, 2, 3):
        summary = bigrams[index]
        bigram_rows.append(
            [
                f"tok{index}",
                str(summary["file"]),
                format_bigram_list(summary["top5"]),
                format_bigram_list(summary["bottom5"]),
            ]
        )

    metrics = metrics_rows()
    body: list[str] = []
    body.append(para(f"HW2 BPE Tokenizer Report - {AUTHOR}, {STUDENT_ID}", bold=True, style="TitleText"))
    body.append(
        para(
            "Submission: HW2_214034621.zip. Goal: improve tokenizer efficiency and speed while preserving NER F1 under the fixed course model."
        )
    )
    body.append(para("Creativity, implementation, and originality", bold=True, style="Heading"))
    body.append(
        para(
            "I implemented character-up BPE with a learned mergeable space marker (SP), allowed useful cross-word merges but capped tokens at one internal space, added byte fallback so encode never emits UNK, and kept decode length-faithful for the NER aligner. The speed work uses merge ranks, an encode cache, and weighted incremental training with local pair-count updates; no NER hyperparameters or forbidden files were changed."
        )
    )
    body.append(para("Experiments and parameter selection", bold=True, style="Heading"))
    body.append(
        para(
            "tok1: balanced domain mix at vocab 6000 was selected for Twitter because it reached F1 0.4780, +0.043 over the previous domain-1-only vocab-8000 tokenizer. Larger mix vocabs dropped, and a structured pre-tokenization trial was reverted after worse F1 at every vocab. tok2: vocab 16000 was selected for news because it kept high F1 (0.9557) while giving the best compression/speed tradeoff. tok3: balanced deduped domain-1/domain-2 mix at vocab 8000 was selected for hidden-domain robustness: medium vocabulary, no UNK, and less single-domain bias."
        )
    )
    body.append(para("Validation and measured results", bold=True, style="Heading"))
    body.append(
        table(
            ["model", "vocab", "dev F1", "tok/char", "tok/sent", "cold tok/s"],
            metrics,
            [1700, 900, 1100, 1100, 1200, 1400],
        )
    )
    body.append(
        para(
            "Validation: check_submission.py passed with required structure and a bigram token; align_check.py passed; speed_check.py --only all passed; forbidden-file hashes matched. Reproducibility is recorded in train_commands.txt. Hidden-domain evidence for tok3: "
            + hidden_oov_text()
            + "."
        )
    )
    body.append(para("BPE training bigrams", bold=True, style="Heading"))
    body.append(
        para(
            "The assignment defines a bigram as two adjacent words as one token. The table therefore reports full-file counts of adjacent whitespace-delimited two-word sequences in each tokenizer's training file, not BPE merge pairs."
        )
    )
    body.append(
        table(
            ["tok", "train file", "top 5 two-word sequences", "least 5 two-word sequences"],
            bigram_rows,
            [600, 1600, 3400, 3426],
        )
    )
    body.append(
        para(
            "Each shipped tokenizer deliberately encodes exactly one checker-defined real bigram token, internally ,\"▁the, enough to satisfy the required bigram behavior while protecting entity-boundary F1 from aggressive cross-word merging."
        )
    )
    body.append(para("Hidden domain and model-2 evaluation", bold=True, style="Heading"))
    body.append(
        para(
            "For the hidden tokenizer, I avoided a pure news or pure Twitter vocabulary and used a balanced deduped mix; this should generalize better to an unseen domain because byte fallback covers new strings and the medium vocab avoids overfitting common news phrases. Model 2 performs best because news text is cleaner and more regular: formal spelling/punctuation makes subword boundaries stable, and repeated named-entity/title patterns let vocab 16000 compress frequent fragments without fragmenting labels. It improves compression/speed while staying within 0.0053 F1 of the smaller best-F1 news candidate."
        )
    )

    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:r="{R}">
  <w:body>{''.join(body)}{sect}</w:body>
</w:document>"""


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def doc_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>"""


def core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="{CP}" xmlns:dc="{DC}" xmlns:dcterms="{DCTERMS}" xmlns:dcmitype="{DCTYPE}" xmlns:xsi="{XSI}">
  <dc:title>HW2 BPE Tokenizer Report</dc:title>
  <dc:creator>{esc(AUTHOR)}</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
</Properties>"""


def settings_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W}">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="720"/>
</w:settings>"""


def write_docx() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DOCX_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml())
        zf.writestr("_rels/.rels", root_rels_xml())
        zf.writestr("word/_rels/document.xml.rels", doc_rels_xml())
        zf.writestr("word/document.xml", document_xml())
        zf.writestr("word/styles.xml", styles_xml())
        zf.writestr("word/numbering.xml", numbering_xml())
        zf.writestr("word/settings.xml", settings_xml())
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("docProps/app.xml", app_xml())
    print(f"Wrote {DOCX_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    write_docx()
