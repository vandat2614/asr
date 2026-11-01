#!/usr/bin/env python3
"""Run the WER normalizer over all train transcripts and report leftovers.

Prints aggregate counts and a few isolated words only (no sentences).
Usage: python3 scripts/check_normalizer.py <folder that holds train/>
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr.normalize import normalize_basic, normalize_text  # noqa: E402

ROOT = Path(sys.argv[1])
with open(ROOT / "train" / "train.jsonl", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

texts = []
for it in items:
    texts.append(it["audio_text"])
    for turn in it.get("history", []):
        if isinstance(turn, dict) and turn.get("audio") and turn.get("audio_text"):
            texts.append(turn["audio_text"])

norm = [normalize_text(t) for t in texts]
basic = [normalize_basic(t) for t in texts]
n = len(texts)

not_idempotent = sum(1 for x in norm if normalize_text(x) != x)
empty = sum(1 for x in norm if not x)
changed_vs_basic = sum(1 for a, b in zip(norm, basic) if a != b)
digit_left = sum(1 for x in norm if re.search(r"\d", x))

tokens = Counter(w for x in norm for w in x.split())
total_tokens = sum(tokens.values())
non_vi_letters = Counter()
for w, c in tokens.items():
    if re.search(r"[wzjf]", w):
        non_vi_letters[w] = c
odd_chars = Counter(
    ch for w in tokens for ch in w if not (ch.isalpha() and "LATIN" in unicodedata.name(ch, ""))
)

print(f"utterances={n} empty_after_normalize={empty}")
print(f"idempotence_failures={not_idempotent}")
print(f"differ_from_basic_normalization={changed_vs_basic} ({100 * changed_vs_basic / n:.1f}%)")
print(f"utterances_with_digits_left={digit_left}")
print(f"tokens={total_tokens} vocab={len(tokens)} hapax(vocab_count_1)={sum(1 for c in tokens.values() if c == 1)}")
print(f"non_latin_or_odd_characters={odd_chars.most_common(10)}")
print(f"tokens_with_w_z_j_f: distinct={len(non_vi_letters)} total={sum(non_vi_letters.values())}")
print(f"top_such_tokens={non_vi_letters.most_common(15)}")
