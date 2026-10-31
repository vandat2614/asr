#!/usr/bin/env python3
"""Print basic statistics of the LifeMate data.

Only reads <split>.jsonl and wav headers. Never touches *_environment/.
Usage: python3 scripts/data_stats.py <folder that holds train/ and public_test/>
"""
import json
import sys
import wave
from pathlib import Path

ROOT = Path(sys.argv[1])


def duration(path):
    """Return the duration of a wav file in seconds, or None if unreadable."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


def pct(sorted_values, q):
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


for split in ("train", "public_test"):
    base = ROOT / split
    with open(base / f"{split}.jsonl", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    wavs = sorted((base / "audio").glob("*/*.wav"))
    durs = [d for d in (duration(p) for p in wavs) if d is not None]
    durs.sort()
    missing = sum(1 for it in items if not (base / it["audio"]).exists())

    with_history = sum(1 for it in items if it.get("history"))
    history_user_turns = sum(
        1
        for it in items
        for turn in it.get("history", [])
        if isinstance(turn, dict) and turn.get("audio")
    )
    print(f"[{split}] samples={len(items)} with_history={with_history}")
    print(f"  wav_files={len(wavs)} unreadable={len(wavs) - len(durs)} missing_final_audio={missing}")
    print(f"  history_user_turns={history_user_turns} utterances_expected={len(items) + history_user_turns}")
    print(
        f"  duration_s: total_h={sum(durs) / 3600:.2f} mean={sum(durs) / len(durs):.2f} "
        f"p50={pct(durs, 0.5):.2f} p95={pct(durs, 0.95):.2f} max={durs[-1]:.2f}"
    )

    if split == "train":
        final_text = sum(1 for it in items if it.get("audio_text"))
        history_text = sum(
            1
            for it in items
            for turn in it.get("history", [])
            if isinstance(turn, dict) and turn.get("audio") and turn.get("audio_text")
        )
        print(f"  labeled_final={final_text} labeled_history_user_turns={history_text}")
