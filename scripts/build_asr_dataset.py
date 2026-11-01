#!/usr/bin/env python3
"""Build the dataset used for ASR fine-tuning: audio.zip + manifest.jsonl.

Reads only train.jsonl and the wav files listed in it (never *_environment/).
Each manifest row is one utterance:
  uid, sample, final, text, audio (path inside the zip),
  primary ("train"|"val"), strict ("train"|"val"), held_out (in either val set).

Usage: python3 scripts/build_asr_dataset.py <folder that holds train/> --username <kaggle username>
Output: outputs/kaggle_dataset/
"""
import argparse
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--username", required=True, help="Kaggle username that owns the dataset")
ap.add_argument("--slug", default="zalo-lifemate-asr")
args = ap.parse_args()


def load_ids(path):
    return set(json.loads(Path(path).read_text(encoding="utf-8")))


primary_val = load_ids(REPO / "splits" / "val_ids.json")
strict_val = load_ids(REPO / "splits" / "strict" / "val_ids.json")

rows = []
samples = set()
with open(args.root / "train" / "train.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        samples.add(item["id"])
        turns = [(item["audio"], item["audio_text"], True)]
        for turn in item.get("history", []):
            if isinstance(turn, dict) and turn.get("audio") and turn.get("audio_text"):
                turns.append((turn["audio"], turn["audio_text"], False))
        for rel_path, text, final in turns:
            uid = Path(rel_path).stem
            rows.append(
                {
                    "uid": uid,
                    "sample": item["id"],
                    "final": final,
                    "text": text,
                    "audio": f"audio/{uid}.wav",
                    "primary": "val" if item["id"] in primary_val else "train",
                    "strict": "val" if item["id"] in strict_val else "train",
                    "held_out": item["id"] in primary_val or item["id"] in strict_val,
                    "_src": args.root / "train" / rel_path,
                }
            )
rows.sort(key=lambda r: r["uid"])

out = REPO / "outputs" / "kaggle_dataset"
out.mkdir(parents=True, exist_ok=True)

total_bytes = 0
with zipfile.ZipFile(out / "audio.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for r in rows:
        src = r.pop("_src")
        z.write(src, arcname=r["audio"])
        total_bytes += src.stat().st_size

with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

metadata = {
    "title": "Zalo LifeMate ASR",
    "id": f"{args.username}/{args.slug}",
    "licenses": [{"name": "other"}],
}
(out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

held_out_samples = len(primary_val | strict_val)
clean_samples = len(samples) - held_out_samples
clean_utts = sum(1 for r in rows if not r["held_out"])
print(f"utterances={len(rows)} samples={len(samples)} wav_mb={total_bytes / 1e6:.0f}")
print(f"primary_val={len(primary_val)} strict_val={len(strict_val)} overlap={len(primary_val & strict_val)}")
print(f"held_out_samples={held_out_samples} clean_train_samples={clean_samples} clean_train_utterances={clean_utts}")
print(f"zip_mb={(out / 'audio.zip').stat().st_size / 1e6:.0f}")
print("output:", out)
