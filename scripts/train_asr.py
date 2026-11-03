#!/usr/bin/env python3
"""Fine-tune a Whisper-style ASR model on the LifeMate train audio (Kaggle).

Expects the Kaggle dataset (audio.zip + manifest.jsonl) and the modules
asr.normalize / asr.metrics on PYTHONPATH. See ADR-0006.

Experiment mode (default): train on utterances with held_out == false and
evaluate after every epoch on both validation sets; keep the best checkpoint.
Final mode (--final): train on all utterances for a fixed number of epochs.
"""
import argparse
import json
import math
import random
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
)

from asr.metrics import corpus_cer, corpus_wer
from asr.normalize import normalize_basic, normalize_text


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/kaggle/input/zalo-lifemate-asr")
    ap.add_argument("--out", default="/kaggle/working/run")
    ap.add_argument("--model", default="Qualcomm-AI-Research/PhoASR-whisper-small")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--spec-augment", action="store_true")
    ap.add_argument("--mask-time-prob", type=float, default=0.05)
    ap.add_argument("--mask-feature-prob", type=float, default=0.0)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-eval", type=int, default=0)
    return ap.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_wav(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sr}: {path}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def prepare_audio(data_dir, work_dir):
    """Return a folder that contains audio/<uid>.wav."""
    if (data_dir / "audio").is_dir():
        return data_dir
    zip_path = data_dir / "audio.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"no audio/ folder or audio.zip in {data_dir}: {sorted(p.name for p in data_dir.iterdir())}")
    target = work_dir / "data"
    if not (target / "audio").is_dir():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target)
    return target


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, rows, audio_root, processor):
        self.rows = rows
        self.audio_root = audio_root
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio = load_wav(self.audio_root / r["audio"])
        feats = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        text = unicodedata.normalize("NFC", r["text"]).strip()
        labels = self.processor.tokenizer(text).input_ids
        return {"input_features": feats, "labels": labels}


def make_collate(processor, decoder_start):
    def collate(batch):
        feats = torch.tensor(np.stack([b["input_features"] for b in batch]))
        padded = processor.tokenizer.pad({"input_ids": [b["labels"] for b in batch]}, return_tensors="pt")
        labels = padded["input_ids"].masked_fill(padded["attention_mask"].ne(1), -100)
        if (labels[:, 0] == decoder_start).all():
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}

    return collate


@torch.inference_mode()
def transcribe(model, processor, audios, device, batch_size):
    model.eval()
    order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
    texts = [""] * len(audios)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        feats = processor.feature_extractor(
            [audios[i] for i in idx], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(
                feats,
                language="vi",
                task="transcribe",
                num_beams=1,
                do_sample=False,
                max_new_tokens=200,
            )
        for i, t in zip(idx, processor.batch_decode(out, skip_special_tokens=True)):
            texts[i] = t.strip()
    return texts


def score(rows, hyps):
    result = {}
    for name in ("primary", "strict"):
        idx = [i for i, r in enumerate(rows) if r[name] == "val"]
        if not idx:
            continue
        refs = [rows[i]["text"] for i in idx]
        hs = [hyps[i] for i in idx]
        result[name] = {
            "utts": len(idx),
            "wer": corpus_wer(refs, hs, normalize_text),
            "wer_raw": corpus_wer(refs, hs, normalize_basic),
            "cer": corpus_cer(refs, hs, normalize_text),
        }
    return result


def save(model, processor, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    data_dir = Path(args.data_dir)
    audio_root = prepare_audio(data_dir, Path("/kaggle/working"))
    rows = load_manifest(data_dir / "manifest.jsonl")

    if args.final:
        train_rows, eval_rows = rows, []
    else:
        train_rows = [r for r in rows if not r["held_out"]]
        eval_rows = [r for r in rows if r["primary"] == "val" or r["strict"] == "val"]
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_eval:
        eval_rows = eval_rows[: args.limit_eval]
    print(f"train_utterances={len(train_rows)} eval_utterances={len(eval_rows)} final={args.final}", flush=True)

    processor = WhisperProcessor.from_pretrained(args.model)
    processor.tokenizer.set_prefix_tokens(language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    if args.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = args.mask_time_prob
        model.config.mask_feature_prob = args.mask_feature_prob
        print(f"spec_augment time={args.mask_time_prob} feature={args.mask_feature_prob}", flush=True)
    model.config.use_cache = False
    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params={n_trainable / 1e6:.1f}M", flush=True)

    eval_audios = [load_wav(audio_root / r["audio"]) for r in eval_rows]

    collate = make_collate(processor, model.config.decoder_start_token_id)
    gen = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        TrainSet(train_rows, audio_root, processor),
        batch_size=args.batch_size,
        shuffle=True,
        generator=gen,
        num_workers=args.workers,
        collate_fn=collate,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")

    history = []
    best = float("inf")
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, count = 0.0, 0
        for step, batch in enumerate(loader):
            feats = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_features=feats, labels=labels).loss
            scaler.scale(loss / args.accum).backward()
            running += loss.item()
            count += 1
            if (step + 1) % args.accum == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if (step + 1) % 50 == 0:
                print(f"epoch={epoch} step={step + 1}/{len(loader)} loss={running / count:.4f} elapsed={time.perf_counter() - start:.0f}s", flush=True)

        record = {"epoch": epoch, "train_loss": running / max(count, 1), "elapsed_s": time.perf_counter() - start}
        if eval_rows:
            hyps = transcribe(model, processor, eval_audios, device, args.eval_batch)
            record["val"] = score(eval_rows, hyps)
            selection = float(np.mean([m["wer"] for m in record["val"].values()]))
            record["selection_wer"] = selection
            if selection < best:
                best = selection
                save(model, processor, out / "best")
                record["saved_best"] = True
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.final:
        save(model, processor, out / "final")
        print("saved final model to", out / "final", flush=True)
    else:
        print("best selection WER:", best, flush=True)


if __name__ == "__main__":
    main()
