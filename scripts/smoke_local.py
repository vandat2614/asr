"""Run LocalASR (CPU by default) on a few validation utterances and compare with stored GPU hypotheses."""
import argparse
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from asr.local import LocalASR  # noqa: E402
from asr.normalize import normalize_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--tag", required=True, help="directory name under outputs/asr/ holding the GPU hypotheses")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--device", default="cpu")
ap.add_argument("--num-beams", type=int, default=1)
args = ap.parse_args()

hyps = [json.loads(line) for line in open(REPO / "outputs" / "asr" / args.tag / "hyps.jsonl", encoding="utf-8")]
audio_rel = {}
for line in open(REPO / "outputs" / "kaggle_dataset" / "manifest.jsonl", encoding="utf-8"):
    r = json.loads(line)
    audio_rel[r["uid"]] = r["audio"]

step = max(1, len(hyps) // args.n)
picked = hyps[::step][: args.n]

with zipfile.ZipFile(REPO / "outputs" / "kaggle_dataset" / "audio.zip") as zf, tempfile.TemporaryDirectory() as tmp:
    names = set(zf.namelist())
    paths = []
    for r in picked:
        rel = audio_rel[r["uid"]]
        member = rel if rel in names else Path(rel).name
        paths.append(zf.extract(member, tmp))

    asr = LocalASR(args.ckpt, device=args.device, num_beams=args.num_beams)
    t0 = time.time()
    local = asr.transcribe(paths)
    print(f"transcribed {len(paths)} files in {time.time() - t0:.1f}s on {args.device}")

same = 0
for r, text in zip(picked, local):
    ok = normalize_text(text) == normalize_text(r["hyp"])
    same += ok
    print(f"\n[{'same' if ok else 'DIFF'}] {r['uid']}")
    print("  ref  :", r["ref"][:120])
    print("  gpu  :", r["hyp"][:120])
    print("  local:", text[:120])
print(f"\nidentical after normalization: {same}/{len(picked)}")
