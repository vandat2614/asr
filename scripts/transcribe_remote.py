#!/usr/bin/env python3
"""Transcribe all utterances of the validation samples with the remote ASR
service and save the raw hypotheses. Resumable: finished utterances are skipped.

Usage: python3 scripts/transcribe_remote.py <folder that holds train/>
           [--chunk 64] [--limit N]
Output: outputs/asr/<model tag>/hyps.jsonl and meta.json
Needs ASR_URL and ASR_TOKEN in the environment.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from asr.remote import RemoteASR  # noqa: E402


def load_ids(path):
    return set(json.loads(Path(path).read_text(encoding="utf-8")))


ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

val_ids = load_ids(REPO / "splits" / "val_ids.json") | load_ids(
    REPO / "splits" / "strict" / "val_ids.json"
)

rows = []
with open(args.root / "train" / "train.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        if item["id"] not in val_ids:
            continue
        rows.append(
            {
                "uid": Path(item["audio"]).stem,
                "sample": item["id"],
                "final": True,
                "path": item["audio"],
                "ref": item["audio_text"],
            }
        )
        for turn in item.get("history", []):
            if isinstance(turn, dict) and turn.get("audio") and turn.get("audio_text"):
                rows.append(
                    {
                        "uid": Path(turn["audio"]).stem,
                        "sample": item["id"],
                        "final": False,
                        "path": turn["audio"],
                        "ref": turn["audio_text"],
                    }
                )
rows.sort(key=lambda r: r["uid"])
if args.limit:
    rows = rows[: args.limit]

asr = RemoteASR()
model = asr.health()["model"]
tag = re.sub(r"[^A-Za-z0-9._-]+", "__", model)
out_dir = REPO / "outputs" / "asr" / tag
out_dir.mkdir(parents=True, exist_ok=True)
hyps_path = out_dir / "hyps.jsonl"
meta_path = out_dir / "meta.json"

done = set()
if hyps_path.exists():
    with open(hyps_path, encoding="utf-8") as f:
        done = {json.loads(line)["uid"] for line in f if line.strip()}
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
else:
    meta = {"model": model, "utterances": 0, "gpu_seconds": 0.0}

todo = [r for r in rows if r["uid"] not in done]
print(f"model={model} total={len(rows)} done={len(done)} todo={len(todo)}")

start = time.perf_counter()
with open(hyps_path, "a", encoding="utf-8") as out:
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i : i + args.chunk]
        before = asr.gpu_seconds
        texts = asr.transcribe([args.root / "train" / r["path"] for r in chunk])
        for r, hyp in zip(chunk, texts):
            record = {k: r[k] for k in ("uid", "sample", "final", "ref")}
            record["hyp"] = hyp
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        meta["utterances"] += len(chunk)
        meta["gpu_seconds"] += asr.gpu_seconds - before
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{min(i + args.chunk, len(todo))}/{len(todo)} elapsed={time.perf_counter() - start:.0f}s")
print("finished:", out_dir)
