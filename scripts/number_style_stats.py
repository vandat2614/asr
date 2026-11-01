#!/usr/bin/env python3
"""Count how times, money and numbers are written in the train transcripts.

Prints only counts and digit-masked shapes (every digit becomes 'D'),
never transcript text. Used for the WER normalization decision.

Usage: python3 scripts/number_style_stats.py <folder that holds train/>
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1])
NUM = r"(?:không|một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|mười|mươi|lăm|rưỡi)"

PATTERNS = {
    "time: D giờ": r"\d+\s*giờ",
    "time: D giờ D (phút)": r"\d+\s*giờ\s*\d+(?:\s*phút)?",
    "time: DhDD (no space)": r"\b\d+h\d*\b",
    "time: D:DD": r"\b\d{1,2}:\d{2}\b",
    "time: D.DD": r"\b\d{1,2}\.\d{2}\b",
    "time: number-word + giờ": rf"\b{NUM}\s+giờ\b",
    "money: D nghìn": r"\d+\s*nghìn",
    "money: D ngàn": r"\d+\s*ngàn",
    "money: D triệu": r"\d+\s*triệu",
    "money: Dk": r"\b\d+\s?k\b",
    "money: D đồng/đ/vnd": r"\d+\s*(?:đồng|đ|vnd)\b",
    "money: trăm nghìn": r"\btrăm nghìn\b",
    "money: trăm ngàn": r"\btrăm ngàn\b",
    "money: thousands separator (D.DDD)": r"\b\d{1,3}(?:[.,]\d{3})+\b",
    "temp: D độ": r"\d+\s*độ",
    "temp: number-word + độ": rf"\b{NUM}\s+độ\b",
}
WORD_COUNTS = {"giờ": r"\bgiờ\b", "nghìn": r"\bnghìn\b", "ngàn": r"\bngàn\b", "độ": r"\bđộ\b"}

with open(ROOT / "train" / "train.jsonl", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

user_texts, assistant_texts = [], []
for it in items:
    user_texts.append(it["audio_text"])
    for turn in it.get("history", []):
        if not isinstance(turn, dict):
            continue
        if turn.get("audio") and turn.get("audio_text"):
            user_texts.append(turn["audio_text"])
        elif turn.get("text") and not turn.get("audio"):
            assistant_texts.append(turn["text"])

n = len(user_texts)
print(f"user_utterances={n} assistant_history_texts={len(assistant_texts)}")

print("\n== user utterances: utterances containing each pattern, top digit-masked shapes ==")
for name, pat in PATTERNS.items():
    rx = re.compile(pat)
    utt, shapes = 0, Counter()
    for t in user_texts:
        found = [m.group(0) for m in rx.finditer(t.lower())]
        if found:
            utt += 1
            shapes.update(re.sub(r"\d", "D", s) for s in found)
    print(f"{name}: {utt} ({100 * utt / n:.1f}%) shapes={shapes.most_common(4)}")

print("\n== user utterances: word and token counts ==")
for name, pat in WORD_COUNTS.items():
    rx = re.compile(pat)
    print(f"occurrences of '{name}': {sum(len(rx.findall(t.lower())) for t in user_texts)}")
digit_tokens = sum(len(re.findall(r"\b\d+\b", t)) for t in user_texts)
numword_tokens = sum(len(re.findall(rf"\b{NUM}\b", t.lower())) for t in user_texts)
print(f"digit tokens={digit_tokens} number-word tokens(proxy, includes 'năm','không','ba')={numword_tokens}")

print("\n== assistant history texts: thousands separator (D.DDD) ==")
rx = re.compile(PATTERNS["money: thousands separator (D.DDD)"])
hits = sum(1 for t in assistant_texts if rx.search(t))
print(f"texts containing it: {hits} of {len(assistant_texts)}")
