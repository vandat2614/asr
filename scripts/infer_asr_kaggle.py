"""Transcribe the validation utterances with a fine-tuned checkpoint (Kaggle kernel).

Inputs are discovered under /kaggle/input:
  - manifest.jsonl (and its audio) from the competition-data dataset
  - a directory holding model.safetensors (the fine-tuned checkpoint)
Outputs go to /kaggle/working: hyps.jsonl and meta.json.
"""
import glob
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

BATCH = 16
SR = 16000
BASE_MODEL = "Qualcomm-AI-Research/PhoASR-whisper-small"
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


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(wav)


def main():
    manifest_path = find_one("/kaggle/input/**/manifest.jsonl")
    ckpt_dir = os.path.dirname(find_one("/kaggle/input/**/model.safetensors"))
    data_root = os.path.dirname(manifest_path)
    print("MANIFEST:", manifest_path, flush=True)
    print("CKPT:", ckpt_dir, flush=True)
    tag_file = os.path.join(ckpt_dir, "tag.txt")
    tag = open(tag_file).read().strip() if os.path.exists(tag_file) else "finetuned/unknown"
    print("TAG:", tag, flush=True)

    rows = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    rows = [r for r in rows if r.get("primary") == "val" or r.get("strict") == "val"]
    print("utterances:", len(rows), flush=True)

    try:
        processor = AutoProcessor.from_pretrained(ckpt_dir)
        processor.feature_extractor  # make sure a feature extractor is present
    except Exception as exc:  # noqa: BLE001
        print("processor fallback to base model:", repr(exc), flush=True)
        processor = AutoProcessor.from_pretrained(BASE_MODEL)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(ckpt_dir)
    if device == "cuda":
        model = model.half()
    model = model.to(device).eval()

    paths = [resolve_audio(data_root, r["audio"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: os.path.getsize(paths[i]))
    hyps = [None] * len(rows)

    t0 = time.time()
    for start in range(0, len(order), BATCH):
        idx = order[start:start + BATCH]
        audio = [load_audio(paths[i]) for i in idx]
        feats = processor.feature_extractor(
            audio, sampling_rate=SR, return_tensors="pt"
        ).input_features.to(device)
        if device == "cuda":
            feats = feats.half()
        with torch.inference_mode():
            ids = model.generate(
                input_features=feats,
                language="vi",
                task="transcribe",
                num_beams=1,
                do_sample=False,
                max_new_tokens=200,
            )
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        for i, text in zip(idx, texts):
            hyps[i] = text.strip()
        done = start + len(idx)
        if (start // BATCH) % 10 == 0:
            print(f"{done}/{len(rows)} elapsed={time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(OUT_DIR, "hyps.jsonl"), "w", encoding="utf-8") as f:
        for r, h in zip(rows, hyps):
            rec = {
                "uid": r["uid"],
                "sample": r["sample"],
                "final": r["final"],
                "ref": r["text"],
                "hyp": h,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {
        "model": tag,
        "utterances": len(rows),
        "gpu_seconds": time.time() - t0,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done", meta, flush=True)


if __name__ == "__main__":
    main()
