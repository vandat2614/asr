#!/usr/bin/env python3
"""Smoke test of the remote ASR service on a few validation utterances.

Usage: python3 scripts/smoke_remote.py <folder that holds train/> [--n 20]
Needs ASR_URL and ASR_TOKEN in the environment.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from asr.metrics import corpus_cer, corpus_wer  # noqa: E402
from asr.normalize import normalize_basic, normalize_text  # noqa: E402
from asr.remote import RemoteASR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--n", type=int, default=20)
args = ap.parse_args()

val_ids = json.loads((REPO / "splits" / "val_ids.json").read_text(encoding="utf-8"))
wanted = set(val_ids[: args.n])
items = []
with open(args.root / "train" / "train.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            if item["id"] in wanted:
                items.append(item)

paths = [args.root / "train" / it["audio"] for it in items]
refs = [it["audio_text"] for it in items]

asr = RemoteASR()
print("health:", asr.health())
t0 = time.perf_counter()
hyps = asr.transcribe(paths)
wall = time.perf_counter() - t0

n = len(items)
print(f"utterances={n} wall_s={wall:.1f} server_gpu_s={asr.gpu_seconds:.1f} gpu_s_per_utt={asr.gpu_seconds / n:.3f}")
print(f"WER primary={corpus_wer(refs, hyps, normalize_text):.4f} raw={corpus_wer(refs, hyps, normalize_basic):.4f}")
print(f"CER primary={corpus_cer(refs, hyps, normalize_text):.4f}")
for ref, hyp in list(zip(refs, hyps))[:5]:
    print("REF:", ref)
    print("HYP:", hyp)
