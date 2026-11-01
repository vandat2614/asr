#!/usr/bin/env python3
"""Show reference/hypothesis pairs where a given reference word was wrong.

Usage: python3 scripts/show_examples.py --word mươi [--n 8]
           [--model-tag Qualcomm-AI-Research__PhoASR-whisper-small]
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from asr.metrics import align_ops  # noqa: E402
from asr.normalize import normalize_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--word", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--model-tag", default="Qualcomm-AI-Research__PhoASR-whisper-small")
args = ap.parse_args()

val_ids = set(json.loads((REPO / "splits" / "val_ids.json").read_text(encoding="utf-8")))
with open(REPO / "outputs" / "asr" / args.model_tag / "hyps.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

shown = 0
for r in rows:
    if r["sample"] not in val_ids:
        continue
    ops = align_ops(normalize_text(r["ref"]).split(), normalize_text(r["hyp"]).split())
    if any(op in ("sub", "del") and rt == args.word for op, rt, _ in ops):
        print(r["uid"])
        print("  REF:", r["ref"])
        print("  HYP:", r["hyp"])
        shown += 1
        if shown >= args.n:
            break
print(f"shown={shown}")
