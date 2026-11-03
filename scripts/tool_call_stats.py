"""Aggregate statistics of the reference tool calls in train.jsonl (labels only)."""
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(line) for line in open(root / "train" / "train.jsonl", encoding="utf-8")]

tools = Counter()
per_sample = Counter()
queries = Counter()
with_cid = with_cid_and_search = 0

for r in rows:
    calls = r["label"]["tool_calls"]
    per_sample[min(len(calls), 5)] += 1
    names = [c["name"] for c in calls]
    tools.update(names)
    has_search = any("contacts" in n and "search" in n for n in names)
    for c in calls:
        if "contacts" in c["name"] and "search" in c["name"]:
            queries[str((c.get("arguments") or {}).get("query"))] += 1
    if any("contact_id" in (c.get("arguments") or {}) for c in calls):
        with_cid += 1
        with_cid_and_search += has_search

print(f"samples={len(rows)}")
print("calls per sample (5 = 5 or more):", dict(sorted(per_sample.items())))
print("\ntop tools:")
for name, n in tools.most_common(30):
    print(f"  {n:5d}  {name}")
print(f"\nsamples with a contact_id argument: {with_cid}, of which also call contacts.search: {with_cid_and_search}")
print("\nmost common contacts.search queries:")
for q, n in queries.most_common(15):
    print(f"  {n:4d}  {q}")
