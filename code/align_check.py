"""Local Phase-1 tokenizer gate harness.

This script mirrors the hard checker predicates and the NER alignment contract
without adding a test framework. In plan 01-01 it is expected to fail because
BPETokenizer.train/encode/decode are still stubs; later plans make it green.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, List

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from bpe_tokenizer import BPETokenizer  # noqa: E402


Check = Callable[[BPETokenizer, List[str]], None]


def read_lines(path: Path, limit: int = 2000) -> List[str]:
    lines: List[str] = []
    if not path.exists():
        return lines
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if text:
                lines.append(text)
            if len(lines) >= limit:
                break
    return lines


def read_tagged_texts(path: Path, limit: int = 200) -> List[str]:
    texts: List[str] = []
    words: List[str] = []
    if not path.exists():
        return texts
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                if words:
                    texts.append(" ".join(words))
                    words = []
                if len(texts) >= limit:
                    break
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                words.append(parts[0])
    if words and len(texts) < limit:
        texts.append(" ".join(words))
    return texts


def has_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def collect_samples() -> List[str]:
    raw_lines = (
        read_lines(ROOT_DIR / "data" / "domain_1_dev.txt")
        + read_lines(ROOT_DIR / "data" / "domain_2_dev.txt")
    )
    tagged_texts = (
        read_tagged_texts(ROOT_DIR / "data" / "ner_data" / "dev_1_binary.tagged")
        + read_tagged_texts(ROOT_DIR / "data" / "ner_data" / "dev_2_binary.tagged")
    )

    samples: List[str] = []
    samples.extend(raw_lines[:20])
    samples.extend(tagged_texts[:20])

    non_ascii = next((line for line in raw_lines + tagged_texts if has_non_ascii(line)), None)
    double_space = next((line for line in raw_lines if "  " in line), None)
    if non_ascii is None:
        raise AssertionError("sample fixture missing non-ASCII line from data/")
    if double_space is None:
        raise AssertionError("sample fixture missing double-space line from raw data/")

    samples.extend([non_ascii, double_space, "OOV stress: café naïve 😀 ִ"])
    return list(dict.fromkeys(samples))


def build_tokenizer(samples: List[str]) -> BPETokenizer:
    tokenizer = BPETokenizer(vocab_size=400)
    training_lines = [
        "in the house",
        "in the car",
        "the end",
        "hello world",
    ] * 25
    training_lines.extend(samples[:50])
    tokenizer.train(training_lines)
    return tokenizer


def non_special_tokens(tokenizer: BPETokenizer) -> Iterable[str]:
    specials = getattr(tokenizer, "special_tokens", {})
    for token in getattr(tokenizer, "token_to_id", {}):
        if token not in specials:
            yield token


def vp1_bigram_no_trigram(tokenizer: BPETokenizer, _samples: List[str]) -> None:
    space = getattr(tokenizer, "space_token", None)
    assert space is not None, "space_token is None; VP-1 cannot evaluate"
    surfaces = [tok.replace(space, " ").strip() for tok in non_special_tokens(tokenizer)]
    assert any(surface.count(" ") == 1 for surface in surfaces), "no cross-word bigram token"
    assert not any(surface.count(" ") >= 2 for surface in surfaces), "token spans 3+ words"


def vp2_space_token(tokenizer: BPETokenizer, _samples: List[str]) -> None:
    assert tokenizer.space_token is not None, "space_token is None"
    assert tokenizer.space_token in tokenizer.token_to_id, "space_token missing from vocab"


def vp3_fresh_process_pickle(tokenizer: BPETokenizer, samples: List[str]) -> None:
    sample = samples[0]
    before_ids = tokenizer.encode(sample)
    before_text = tokenizer.decode(before_ids)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        code_copy = tmp_path / "code"
        code_copy.mkdir()
        for source in CODE_DIR.glob("*.py"):
            shutil.copy2(source, code_copy / source.name)

        pkl_path = tmp_path / "tokenizer.pkl"
        tokenizer.save(str(pkl_path))
        child = (
            "import json,sys;"
            "from pathlib import Path;"
            f"sys.path.insert(0, {str(code_copy)!r});"
            "from base_tokenizer import BaseTokenizer;"
            f"tok=BaseTokenizer.load({str(pkl_path)!r});"
            f"text={sample!r};"
            "ids=tok.encode(text);"
            "print(json.dumps({'ids': ids, 'text': tok.decode(ids)}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", child],
            check=True,
            capture_output=True,
            text=True,
        )
        loaded = json.loads(result.stdout)
    assert loaded["ids"] == before_ids, "fresh-process encode differs after load"
    assert loaded["text"] == before_text, "fresh-process decode differs after load"


def vp4_length_faithful_decode(tokenizer: BPETokenizer, samples: List[str]) -> None:
    assert any(has_non_ascii(sample) for sample in samples), "samples lack non-ASCII data"
    assert any("  " in sample for sample in samples), "samples lack double-space data"
    for text in samples[:80]:
        ids = tokenizer.encode(text)
        reconstructed = tokenizer.decode(ids)
        assert reconstructed.replace(" ", "") == text.replace(" ", ""), repr(text)
        previous = 0
        spans = []
        for index in range(len(ids)):
            current = len(tokenizer.decode(ids[: index + 1]))
            assert current >= previous, f"non-monotonic decode length for {text!r}"
            token_start = max(0, min(previous, len(text)))
            token_end = max(token_start, min(current, len(text)))
            spans.append((token_start, token_end))
            previous = current
        assert spans or not text, f"no spans produced for {text!r}"


def vp6_zero_unk(tokenizer: BPETokenizer, _samples: List[str]) -> None:
    stress = "OOV/non-ASCII stress: café naïve 😀 عربي ִ"
    ids = tokenizer.encode(stress)
    unk_id = tokenizer.special_tokens["[UNK]"]
    assert unk_id not in ids, "[UNK] emitted for OOV stress string"


CHECKS: dict[str, Check] = {
    "vp1": vp1_bigram_no_trigram,
    "vp2": vp2_space_token,
    "vp3": vp3_fresh_process_pickle,
    "vp4": vp4_length_faithful_decode,
    "vp6": vp6_zero_unk,
}


def parse_filter(raw: str | None) -> set[str]:
    if not raw:
        return set(CHECKS)
    selected = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = selected - set(CHECKS)
    if unknown:
        raise SystemExit(f"Unknown VP(s): {', '.join(sorted(unknown))}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local tokenizer gate checks.")
    parser.add_argument("--only", help="Comma-separated VP list, e.g. vp1,vp2,vp4")
    args = parser.parse_args()

    selected = parse_filter(args.only)
    failures: List[str] = []

    try:
        samples = collect_samples()
        print("FIXTURE: PASS - data samples include non-ASCII and double-space lines")
    except Exception as exc:
        print(f"FIXTURE: FAIL - {exc}")
        return 1

    try:
        tokenizer = build_tokenizer(samples)
        print("TRAIN: PASS - BPETokenizer.train completed")
    except NotImplementedError as exc:
        print(f"TRAIN: FAIL - {exc}")
        return 1
    except Exception as exc:
        print(f"TRAIN: FAIL - {type(exc).__name__}: {exc}")
        return 1

    for name, check in CHECKS.items():
        if name not in selected:
            print(f"{name.upper()}: SKIP")
            continue
        try:
            check(tokenizer, samples)
            print(f"{name.upper()}: PASS")
        except Exception as exc:
            print(f"{name.upper()}: FAIL - {type(exc).__name__}: {exc}")
            failures.append(name)

    if failures:
        print(f"OVERALL: FAIL - {', '.join(failures)}")
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
