"""List the most frequently missed name tokens for one model, with one example each."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from slot_eval import name_tokens  # noqa: E402
from asr.normalize import normalize_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model_tag")
ap.add_argument("--top", type=int, default=25)
args = ap.parse_args()

path = REPO / "outputs" / "asr" / args.model_tag / "hyps.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8")]

total, missed, example = Counter(), Counter(), {}
for r in rows:
    hyp_tokens = set(normalize_text(r["hyp"]).split())
    for tok in name_tokens(r["ref"]):
        total[tok] += 1
        if tok not in hyp_tokens:
            missed[tok] += 1
            example.setdefault(tok, (r["ref"], r["hyp"]))

print(f"model={args.model_tag} name_tokens={sum(total.values())} missed={sum(missed.values())}")
for tok, n in missed.most_common(args.top):
    ref, hyp = example[tok]
    print(f"\n{tok}: missed {n}/{total[tok]}")
    print(f"  ref: {ref[:110]}")
    print(f"  hyp: {hyp[:110]}")
