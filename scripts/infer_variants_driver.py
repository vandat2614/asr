# --- Kaggle driver: appended after src/asr/local.py (see scripts/build_infer_kernel.sh) ---
import glob  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

VARIANTS = [
    ("greedy-fp16", "fp16", 1),
    ("greedy-fp32", "fp32", 1),
    ("beam5-fp16", "fp16", 5),
    ("beam5-fp32", "fp32", 5),
]
OUT_DIR = "/kaggle/working"


def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def resolve_audio(root, rel):
    # The dataset may be auto-extracted into a nested audio/audio/ directory.
    for cand in (os.path.join(root, rel), os.path.join(root, "audio", rel)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(rel)


def main():
    manifest_path = find_one("/kaggle/input/**/manifest.jsonl")
    ckpt_dir = os.path.dirname(find_one("/kaggle/input/**/model.safetensors"))
    data_root = os.path.dirname(manifest_path)
    tag_file = os.path.join(ckpt_dir, "tag.txt")
    tag = open(tag_file).read().strip() if os.path.exists(tag_file) else "finetuned/unknown"
    print("CKPT:", ckpt_dir, "TAG:", tag, flush=True)

    rows = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    rows = [r for r in rows if r.get("primary") == "val" or r.get("strict") == "val"]
    paths = [resolve_audio(data_root, r["audio"]) for r in rows]
    print("utterances:", len(rows), flush=True)

    for name, precision, beams in VARIANTS:
        asr = LocalASR(ckpt_dir, precision=precision, num_beams=beams)
        t0 = time.time()
        hyps = asr.transcribe(paths)
        seconds = time.time() - t0
        with open(os.path.join(OUT_DIR, f"hyps_{name}.jsonl"), "w", encoding="utf-8") as f:
            for r, h in zip(rows, hyps):
                rec = {"uid": r["uid"], "sample": r["sample"], "final": r["final"], "ref": r["text"], "hyp": h}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta = {"model": f"{tag}/{name}", "utterances": len(rows), "gpu_seconds": seconds}
        with open(os.path.join(OUT_DIR, f"meta_{name}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print("done", name, f"{seconds:.0f}s", flush=True)
        del asr
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
