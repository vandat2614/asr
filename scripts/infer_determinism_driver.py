# --- Kaggle driver: appended after src/asr/local.py (see scripts/build_infer_kernel.sh) ---
import glob  # noqa: E402
import json  # noqa: E402

N_UTTS = 400
CONFIGS = [("b16-run1", 16), ("b16-run2", 16), ("b4", 4), ("b1", 1)]
OUT_DIR = "/kaggle/working"


def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def resolve_audio(root, rel):
    for cand in (os.path.join(root, rel), os.path.join(root, "audio", rel)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(rel)


def main():
    manifest_path = find_one("/kaggle/input/**/manifest.jsonl")
    ckpt_dir = os.path.dirname(find_one("/kaggle/input/**/model.safetensors"))
    data_root = os.path.dirname(manifest_path)

    rows = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    rows = [r for r in rows if r.get("primary") == "val" or r.get("strict") == "val"][:N_UTTS]
    paths = [resolve_audio(data_root, r["audio"]) for r in rows]

    results = {}
    for name, batch_size in CONFIGS:
        asr = LocalASR(ckpt_dir, num_beams=5, batch_size=batch_size, precision="fp16")
        results[name] = asr.transcribe(paths)
        print("done", name, flush=True)
        del asr
        torch.cuda.empty_cache()

    base = results["b16-run1"]
    lines = []
    for name, out in results.items():
        diff = [i for i, (a, b) in enumerate(zip(base, out)) if a != b]
        lines.append(f"{name}: identical to b16-run1 on {len(base) - len(diff)}/{len(base)} utterances")
        for i in diff[:5]:
            lines.append(f"  DIFF {rows[i]['uid']}\n    b16-run1: {base[i]!r}\n    {name}: {out[i]!r}")
    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(OUT_DIR, "determinism.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
