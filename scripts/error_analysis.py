#!/usr/bin/env python3
"""Error analysis of saved ASR hypotheses on the primary validation split.

Reports the error mix, error rate per word category, how many errors fall on
words unseen in the training split, and the most common confusions.

Usage: python3 scripts/error_analysis.py <folder that holds train/>
           [--model-tag Qualcomm-AI-Research__PhoASR-whisper-small] [--top 25]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from asr.metrics import align_ops  # noqa: E402
from asr.normalize import normalize_text  # noqa: E402

NUMBER_WORDS = set("không một hai ba bốn năm sáu bảy tám chín mười mươi trăm nghìn triệu tỷ linh rưỡi".split())
TIME_WORDS = set(
    "giờ phút giây tiếng sáng trưa chiều tối đêm khuya mai nay hôm qua ngày tháng tuần lúc sau trước kém".split()
)

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--model-tag", default="Qualcomm-AI-Research__PhoASR-whisper-small")
ap.add_argument("--top", type=int, default=25)
args = ap.parse_args()


def load_ids(path):
    return set(json.loads(Path(path).read_text(encoding="utf-8")))


train_ids = load_ids(REPO / "splits" / "train_ids.json")
val_ids = load_ids(REPO / "splits" / "val_ids.json")

vocab = Counter()
with open(args.root / "train" / "train.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        if item["id"] not in train_ids:
            continue
        texts = [item["audio_text"]] + [
            t["audio_text"]
            for t in item.get("history", [])
            if isinstance(t, dict) and t.get("audio") and t.get("audio_text")
        ]
        for text in texts:
            vocab.update(normalize_text(text).split())

with open(REPO / "outputs" / "asr" / args.model_tag / "hyps.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
rows = [r for r in rows if r["sample"] in val_ids]

ops_count = Counter()
cat_total = Counter()
cat_err = Counter()
subs, dels, inss = Counter(), Counter(), Counter()


def categories(token):
    cats = []
    if token in NUMBER_WORDS:
        cats.append("number words")
    if token in TIME_WORDS:
        cats.append("time-related words")
    if any(c in token for c in "wzjf"):
        cats.append("foreign-letter words (w/z/j/f)")
    count = vocab[token]
    cats.append("unseen in train" if count == 0 else "rare in train (1-5)" if count <= 5 else "frequent in train (>5)")
    return cats


for r in rows:
    ref = normalize_text(r["ref"]).split()
    hyp = normalize_text(r["hyp"]).split()
    for op, rt, ht in align_ops(ref, hyp):
        ops_count[op] += 1
        if rt is not None:
            for c in categories(rt):
                cat_total[c] += 1
                if op != "ok":
                    cat_err[c] += 1
        if op == "sub":
            subs[(rt, ht)] += 1
        elif op == "del":
            dels[rt] += 1
        elif op == "ins":
            inss[ht] += 1

n_ref = ops_count["ok"] + ops_count["sub"] + ops_count["del"]
n_err = ops_count["sub"] + ops_count["del"] + ops_count["ins"]
print(f"model={args.model_tag} utterances={len(rows)} ref_words={n_ref}")
print(f"WER={n_err / n_ref:.4f} substitutions={ops_count['sub']} deletions={ops_count['del']} insertions={ops_count['ins']}")

print("\nerror rate on reference words by category (substitution or deletion):")
ref_errors = ops_count["sub"] + ops_count["del"]
for c, total in sorted(cat_total.items(), key=lambda kv: -kv[1]):
    print(f"  {c}: words={total} error_rate={cat_err[c] / total:.3f} share_of_all_ref_errors={cat_err[c] / max(ref_errors, 1):.3f}")

print(f"\ntop {args.top} substitutions (ref -> hyp):")
for (rt, ht), c in subs.most_common(args.top):
    print(f"  {c:3d}  {rt} -> {ht}")
print(f"\ntop {args.top} deleted words:")
for w, c in dels.most_common(args.top):
    print(f"  {c:3d}  {w}")
print(f"\ntop {args.top} inserted words:")
for w, c in inss.most_common(args.top):
    print(f"  {c:3d}  {w}")
