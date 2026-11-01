#!/usr/bin/env python3
"""Score every model found under outputs/asr/ on both validation splits.

Prints a markdown table. Usage: python3 scripts/score_asr.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from asr.metrics import corpus_cer, corpus_wer  # noqa: E402
from asr.normalize import normalize_basic, normalize_text  # noqa: E402


def load_ids(path):
    return set(json.loads(Path(path).read_text(encoding="utf-8")))


splits = {
    "primary": load_ids(REPO / "splits" / "val_ids.json"),
    "strict": load_ids(REPO / "splits" / "strict" / "val_ids.json"),
}

print("| model | split | utts | WER | WER raw | CER | WER final-only | GPU s/utt |")
print("|---|---|---|---|---|---|---|---|")
for d in sorted((REPO / "outputs" / "asr").iterdir()):
    if not (d / "hyps.jsonl").exists():
        continue
    with open(d / "hyps.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    gpu_per_utt = meta["gpu_seconds"] / max(meta["utterances"], 1)
    for name, ids in splits.items():
        sub = [r for r in rows if r["sample"] in ids]
        if not sub:
            continue
        refs = [r["ref"] for r in sub]
        hyps = [r["hyp"] for r in sub]
        fin = [r for r in sub if r["final"]]
        wer = corpus_wer(refs, hyps, normalize_text)
        wer_raw = corpus_wer(refs, hyps, normalize_basic)
        cer = corpus_cer(refs, hyps, normalize_text)
        wer_fin = corpus_wer([r["ref"] for r in fin], [r["hyp"] for r in fin], normalize_text)
        print(
            f"| {meta['model']} | {name} | {len(sub)} | {wer:.4f} | {wer_raw:.4f} "
            f"| {cer:.4f} | {wer_fin:.4f} | {gpu_per_utt:.3f} |"
        )
