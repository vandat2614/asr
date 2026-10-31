#!/usr/bin/env python3
"""Create a grouped train/validation split of the LifeMate train set.

Samples whose final transcripts are near-duplicates are merged into one group,
and a group is never split across train and validation. All utterances of a
sample (final request and history turns) stay on the same side.

Only reads train.jsonl. Writes sample ids (no transcripts) to splits/.

Usage:
    python3 scripts/make_split.py <folder that holds train/> [--val-ratio 0.2]
        [--seed 20261101] [--threshold 0.9] [--out splits]
"""
import argparse
import json
import random
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


def normalize(text):
    """Lowercase, strip punctuation and collapse whitespace (grouping only)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def count_utterances(item):
    history = sum(
        1 for t in item.get("history", []) if isinstance(t, dict) and t.get("audio")
    )
    return 1 + history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="folder that holds train/")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", type=Path, default=Path("splits"))
    args = ap.parse_args()

    with open(args.root / "train" / "train.jsonl", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    ids = [it["id"] for it in items]
    texts = [normalize(it["audio_text"]) for it in items]
    n = len(items)

    # Group near-duplicate final transcripts with union-find.
    uf = UnionFind(n)
    sm = SequenceMatcher(None, autojunk=False)
    for i in range(n):
        sm.set_seq2(texts[i])
        for j in range(i + 1, n):
            sm.set_seq1(texts[j])
            if (
                sm.real_quick_ratio() >= args.threshold
                and sm.quick_ratio() >= args.threshold
                and sm.ratio() >= args.threshold
            ):
                uf.union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    group_list = sorted(groups.values(), key=lambda g: g[0])

    # Deterministic shuffle, then fill validation until the target size.
    random.Random(args.seed).shuffle(group_list)
    target = round(n * args.val_ratio)
    val_idx, train_idx = [], []
    for g in group_list:
        (val_idx if len(val_idx) < target else train_idx).extend(g)

    args.out.mkdir(parents=True, exist_ok=True)
    train_ids = sorted(ids[i] for i in train_idx)
    val_ids = sorted(ids[i] for i in val_idx)
    (args.out / "train_ids.json").write_text(json.dumps(train_ids), encoding="utf-8")
    (args.out / "val_ids.json").write_text(json.dumps(val_ids), encoding="utf-8")

    def side_stats(idx):
        utt = sum(count_utterances(items[i]) for i in idx)
        with_hist = sum(1 for i in idx if items[i].get("history"))
        return {"samples": len(idx), "utterances": utt, "with_history": with_hist}

    meta = {
        "seed": args.seed,
        "threshold": args.threshold,
        "val_ratio": args.val_ratio,
        "groups": len(group_list),
        "train": side_stats(train_idx),
        "val": side_stats(val_idx),
    }
    (args.out / "split_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    sizes = sorted((len(g) for g in group_list), reverse=True)
    print(f"samples={n} groups={len(group_list)} singletons={sizes.count(1)}")
    print(f"largest_groups={sizes[:5]}")
    print(f"train={meta['train']}")
    print(f"val={meta['val']}")


if __name__ == "__main__":
    main()
