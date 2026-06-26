"""Hidden-generalization selector for tokenizer 3 (Phase 4, 04-04 / D-04).

Joins tok3 CPU evidence (tuning/results/tok3_cpu_sweep.csv) with visible-domain
VM F1 (tuning/results/tok3_vm_f1_results.csv), hard-gates unsafe candidates, then
scores the survivors by an auditable normalized formula that deliberately does
NOT pick by tokens/char alone:

  score = 0.35*mean_visible_f1        (higher better)
        + 0.25*combined_tokens/char   (lower  better)
        + 0.15*domain_balance         (smaller F1+tok/char gap better)
        + 0.10*oov_robustness         (stress pass-rate)
        + 0.10*reconstruction_gate    (min reconstruction across splits)
        + 0.05*speed                  (higher better)

Tie-break (top scores within 0.01): smaller vocab, then lower combined tok/char,
then higher worst-domain F1, then faster. Writes tuning/results/tok3_selection.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from base_tokenizer import BaseTokenizer

WEIGHTS = {
    "f1": 0.35,
    "tpc": 0.25,
    "balance": 0.15,
    "oov": 0.10,
    "recon": 0.10,
    "speed": 0.05,
}


def load_cpu(path):
    by_vocab = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        v = int(r["vocab_size"])
        by_vocab.setdefault(v, {})[r["eval_split"]] = r
    return by_vocab


def load_vm(path):
    by_vocab = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        v = int(r["vocab_size"])
        by_vocab.setdefault(v, {})[r["domain"]] = r
    return by_vocab


def gate(vocab, cpu_rows, vm_rows, candidate_dir):
    """Return (passed: bool, reason: str)."""
    if set(cpu_rows) < {"domain_1_dev", "domain_2_dev", "combined_visible_dev", "stress_oov"}:
        return False, "missing cpu split rows"
    if set(vm_rows) < {"1", "2"}:
        return False, "missing vm domain rows"
    for split, r in cpu_rows.items():
        if r["reconstruction_rate"] != "1.000000":
            return False, f"reconstruction<1 on {split}"
        if r["unk_count"] != "0":
            return False, f"unk>0 on {split}"
        if int(r["trigram_token_count"]) != 0:
            return False, "trigram token present"
        if int(r["bigram_token_count"]) <= 0:
            return False, "no bigram token"
    stress = cpu_rows["stress_oov"]
    if stress["oov_stress_cases_passed"] != stress["oov_stress_cases_total"]:
        return False, "oov stress incomplete"
    try:
        float(vm_rows["1"]["f1"]); float(vm_rows["2"]["f1"])
    except (KeyError, ValueError):
        return False, "malformed vm f1"
    path = candidate_dir / f"tokenizer_3_vocab{vocab}.pkl"
    if not path.exists():
        return False, "candidate pkl missing"
    BaseTokenizer.load(str(path))  # raises if unloadable
    return True, "ok"


def raw_components(vocab, cpu_rows, vm_rows):
    f1_1 = float(vm_rows["1"]["f1"])
    f1_2 = float(vm_rows["2"]["f1"])
    d1_tpc = float(cpu_rows["domain_1_dev"]["tokens_per_char"])
    d2_tpc = float(cpu_rows["domain_2_dev"]["tokens_per_char"])
    combined = cpu_rows["combined_visible_dev"]
    stress = cpu_rows["stress_oov"]
    min_recon = min(float(cpu_rows[s]["reconstruction_rate"]) for s in cpu_rows)
    return {
        "mean_f1": (f1_1 + f1_2) / 2.0,
        "worst_f1": min(f1_1, f1_2),
        "f1_d1": f1_1,
        "f1_d2": f1_2,
        "combined_tpc": float(combined["tokens_per_char"]),
        "balance": abs(f1_1 - f1_2) + abs(d1_tpc - d2_tpc),
        "oov": int(stress["oov_stress_cases_passed"]) / int(stress["oov_stress_cases_total"]),
        "recon": min_recon,
        "speed": float(combined["tokens_per_second"]),
    }


def norm(value, lo, hi, higher_better):
    if hi == lo:
        return 1.0
    frac = (value - lo) / (hi - lo)
    return frac if higher_better else 1.0 - frac


def main():
    p = argparse.ArgumentParser(description="Score and select tok3 vocab (D-04).")
    p.add_argument("--cpu_csv", default="tuning/results/tok3_cpu_sweep.csv")
    p.add_argument("--vm_csv", default="tuning/results/tok3_vm_f1_results.csv")
    p.add_argument("--output_csv", default="tuning/results/tok3_selection.csv")
    p.add_argument("--candidate_dir", type=Path, default=Path("tuning/tokenizers"))
    args = p.parse_args()

    cpu = load_cpu(args.cpu_csv)
    vm = load_vm(args.vm_csv)
    vocabs = sorted(set(cpu) & set(vm))

    comps, gates = {}, {}
    for v in vocabs:
        passed, reason = gate(v, cpu[v], vm[v], args.candidate_dir)
        gates[v] = (passed, reason)
        if passed:
            comps[v] = raw_components(v, cpu[v], vm[v])

    passing = [v for v in vocabs if gates[v][0]]
    ranges = {}
    for key, higher in [("mean_f1", True), ("combined_tpc", False), ("balance", False),
                        ("oov", True), ("recon", True), ("speed", True)]:
        vals = [comps[v][key] for v in passing]
        ranges[key] = (min(vals), max(vals), higher)

    scored = {}
    for v in passing:
        c = comps[v]
        n = {k: norm(c[k], ranges[k][0], ranges[k][1], ranges[k][2])
             for k in ranges}
        score = (WEIGHTS["f1"] * n["mean_f1"] + WEIGHTS["tpc"] * n["combined_tpc"]
                 + WEIGHTS["balance"] * n["balance"] + WEIGHTS["oov"] * n["oov"]
                 + WEIGHTS["recon"] * n["recon"] + WEIGHTS["speed"] * n["speed"])
        scored[v] = (score, n)

    # selection with tie-breakers
    def tiebreak_key(v):
        score = scored[v][0]
        c = comps[v]
        return (-round(score, 4), v, c["combined_tpc"], -c["worst_f1"], -c["speed"])

    ordered = sorted(passing, key=tiebreak_key)
    # collapse near-ties (within 0.01) to smaller vocab first
    best = ordered[0]
    top_score = scored[best][0]
    near = [v for v in passing if abs(scored[v][0] - top_score) < 0.01]
    if near:
        best = sorted(near, key=lambda v: (v, comps[v]["combined_tpc"],
                                           -comps[v]["worst_f1"], -comps[v]["speed"]))[0]

    fields = ["vocab_size", "gate_status", "gate_reason", "mean_visible_f1",
              "worst_domain_f1", "f1_domain1", "f1_domain2", "combined_tokens_per_char",
              "domain_balance_gap", "oov_robustness", "reconstruction_min", "speed_tok_s",
              "score", "selected"]
    rows = []
    for v in vocabs:
        passed, reason = gates[v]
        if passed:
            c = comps[v]
            rows.append({
                "vocab_size": v, "gate_status": "pass", "gate_reason": reason,
                "mean_visible_f1": f"{c['mean_f1']:.4f}", "worst_domain_f1": f"{c['worst_f1']:.4f}",
                "f1_domain1": f"{c['f1_d1']:.4f}", "f1_domain2": f"{c['f1_d2']:.4f}",
                "combined_tokens_per_char": f"{c['combined_tpc']:.6f}",
                "domain_balance_gap": f"{c['balance']:.6f}", "oov_robustness": f"{c['oov']:.4f}",
                "reconstruction_min": f"{c['recon']:.6f}", "speed_tok_s": f"{c['speed']:.1f}",
                "score": f"{scored[v][0]:.4f}", "selected": str(v == best).lower(),
            })
        else:
            rows.append({f: "" for f in fields} | {
                "vocab_size": v, "gate_status": "fail", "gate_reason": reason, "selected": "false"})

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"SELECTED tok3 vocab = {best}")
    print("vocab |  score | meanF1 | worstF1 | comb_tpc | balance | speed(M/s)")
    for v in vocabs:
        if gates[v][0]:
            c = comps[v]
            print(f"{v:>5} | {scored[v][0]:.4f} | {c['mean_f1']:.4f} | {c['worst_f1']:.4f}  | "
                  f"{c['combined_tpc']:.4f}   | {c['balance']:.4f} | {c['speed']/1e6:.1f}"
                  + ("   <== SELECTED" if v == best else ""))
        else:
            print(f"{v:>5} | GATED OUT: {gates[v][1]}")


if __name__ == "__main__":
    main()
