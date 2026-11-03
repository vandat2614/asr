"""Show final utterances whose number runs are not reproduced exactly."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from slot_eval import number_runs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model_tag")
ap.add_argument("--limit", type=int, default=30)
args = ap.parse_args()

path = REPO / "outputs" / "asr" / args.model_tag / "hyps.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
rows = [r for r in rows if r["final"]]

bad = []
for r in rows:
    missing = Counter(number_runs(r["ref"])) - Counter(number_runs(r["hyp"]))
    if missing:
        bad.append((r, missing))

print(f"model={args.model_tag} final_utts={len(rows)} with_number_errors={len(bad)}")
for r, missing in bad[: args.limit]:
    print("\nmissing:", " | ".join(" ".join(run) for run in missing))
    print("  ref:", r["ref"][:130])
    print("  hyp:", r["hyp"][:130])
