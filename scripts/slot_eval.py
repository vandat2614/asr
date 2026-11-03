"""Slot-critical scoring: exact number runs and name-token recall.

Reads outputs/asr/*/hyps.jsonl (union of both validation sets) and prints one
markdown table. A number run is a maximal sequence of number words after
normalization (digits are expanded, so times are covered).
"""
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from asr.normalize import normalize_text  # noqa: E402

NUM = {
    "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín",
    "mười", "mươi", "trăm", "nghìn", "triệu", "tỷ", "linh", "phẩy",
}
PUNCT = ".,;:!?\"'()"


def number_runs(text):
    runs, cur = [], []
    for tok in normalize_text(text).split():
        if tok in NUM:
            cur.append(tok)
        elif cur:
            runs.append(tuple(cur))
            cur = []
    if cur:
        runs.append(tuple(cur))
    # A lone "không" is almost always the negation, not the number zero.
    return [r for r in runs if set(r) != {"không"}]


def name_tokens(raw):
    toks = raw.split()
    out = []
    for i, tok in enumerate(toks):
        word = tok.strip(PUNCT)
        if i == 0 or not word[:1].isupper():
            continue
        if toks[i - 1].endswith((".", "!", "?")):
            continue
        out += normalize_text(word).split()
    return out


def score(rows):
    n_runs = ok_runs = utt_num = utt_num_ok = n_names = ok_names = 0
    for r in rows:
        ref_runs = Counter(number_runs(r["ref"]))
        hyp_runs = Counter(number_runs(r["hyp"]))
        if ref_runs:
            n_runs += sum(ref_runs.values())
            ok_runs += sum((ref_runs & hyp_runs).values())
            utt_num += 1
            utt_num_ok += not (ref_runs - hyp_runs)
        hyp_tokens = set(normalize_text(r["hyp"]).split())
        names = name_tokens(r["ref"])
        n_names += len(names)
        ok_names += sum(t in hyp_tokens for t in names)
    return n_runs, ok_runs, utt_num, utt_num_ok, n_names, ok_names


def pct(a, b):
    return f"{a / b:.3f}" if b else "n/a"


def main():
    print("| model | subset | number runs | run acc | utts all numbers right | name tokens | name recall |")
    print("|---|---|---|---|---|---|---|")
    for path in sorted((REPO / "outputs" / "asr").glob("*/hyps.jsonl")):
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        for label, subset in (("all", rows), ("final", [r for r in rows if r["final"]])):
            n_runs, ok_runs, utt_num, utt_ok, n_names, ok_names = score(subset)
            print(
                f"| {path.parent.name} | {label} | {n_runs} | {pct(ok_runs, n_runs)} "
                f"| {pct(utt_ok, utt_num)} ({utt_num} utts) | {n_names} | {pct(ok_names, n_names)} |"
            )


if __name__ == "__main__":
    main()
