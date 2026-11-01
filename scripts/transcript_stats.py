#!/usr/bin/env python3
"""Aggregate statistics of the train transcripts (no transcript text is printed).

Used to decide the text normalization for WER (see docs/decisions).
Usage: python3 scripts/transcript_stats.py <folder that holds train/>
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1])
NUMBER_WORDS = {
    "không", "một", "hai", "ba", "bốn", "tư", "năm", "sáu", "bảy", "tám",
    "chín", "mười", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "rưỡi",
}
FOREIGN_LETTERS = set("wzjf")

with open(ROOT / "train" / "train.jsonl", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

texts = []
for it in items:
    texts.append(it["audio_text"])
    for turn in it.get("history", []):
        if isinstance(turn, dict) and turn.get("audio") and turn.get("audio_text"):
            texts.append(turn["audio_text"])

n = len(texts)
punct = Counter()
words_per = []
with_digit = with_upper = with_punct = with_numword = with_foreign = 0
with_time_like = with_nfd = 0

for t in texts:
    if unicodedata.normalize("NFC", t) != t:
        with_nfd += 1
    chars = [c for c in t if not c.isalnum() and not c.isspace()]
    punct.update(chars)
    words = re.findall(r"\w+", t.lower())
    words_per.append(len(words))
    with_digit += any(c.isdigit() for c in t)
    with_upper += any(c.isupper() for c in t)
    with_punct += bool(chars)
    with_numword += any(w in NUMBER_WORDS for w in words)
    with_foreign += any(set(w) & FOREIGN_LETTERS for w in words)
    with_time_like += bool(re.search(r"\d\s*(h|giờ|:)|\d+[:.]\d+", t.lower()))


def pct(x):
    return f"{x} ({100 * x / n:.1f}%)"


print(f"utterances={n}")
print(f"not_in_NFC_form={pct(with_nfd)}")
print(f"with_digits={pct(with_digit)}")
print(f"with_time_like_pattern={pct(with_time_like)}")
print(f"with_number_words(proxy)={pct(with_numword)}")
print(f"with_uppercase={pct(with_upper)}")
print(f"with_punctuation={pct(with_punct)}")
print(f"with_letters_w_z_j_f(foreign-word proxy)={pct(with_foreign)}")
print(f"top_punctuation={punct.most_common(12)}")
print(f"words_per_utterance: mean={sum(words_per) / n:.2f} max={max(words_per)}")
