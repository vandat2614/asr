"""Transcribe audio files with the offline ASR module.

Usage: python3 scripts/transcribe_local.py [--model models/asr-final] a.wav b.wav
Prints one JSON object per file.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from asr.local import LocalASR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("audio", nargs="+")
ap.add_argument("--model", default=str(REPO / "models" / "asr-final"))
ap.add_argument("--device", default=None)
args = ap.parse_args()

asr = LocalASR(args.model, device=args.device)
for path, text in zip(args.audio, asr.transcribe(args.audio)):
    print(json.dumps({"audio": path, "text": text}, ensure_ascii=False))
