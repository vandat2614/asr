#!/usr/bin/env python3
"""Run ASR fine-tuning experiments inside a Kaggle script kernel.

Reads experiments.json from the code dataset: a list of
{"name": ..., "args": [...]} entries whose args are passed to train_asr.py.
Runs one experiment per visible GPU at a time and writes logs and
checkpoints to /kaggle/working.
"""
import glob
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch


def find_dir(filename):
    hits = sorted(glob.glob(f"/kaggle/input/**/{filename}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{filename} not found under /kaggle/input")
    return Path(hits[0]).parent


CODE = find_dir("train_asr.py")
DATA = find_dir("manifest.jsonl")
print("CODE:", CODE, flush=True)
print("DATA:", DATA, flush=True)

# Make asr.normalize and asr.metrics importable.
pkg = Path("/tmp/pkg/asr")
pkg.mkdir(parents=True, exist_ok=True)
for name in ("normalize.py", "metrics.py"):
    shutil.copy(CODE / name, pkg / name)
(pkg / "__init__.py").touch()

# Kaggle may unpack audio.zip into audio/audio/; expose a clean root.
inner = DATA / "audio" / "audio"
audio_dir = inner if inner.is_dir() else DATA / "audio"
if audio_dir.is_dir():
    data_root = Path("/tmp/data_fix")
    data_root.mkdir(exist_ok=True)
    for name, target in (("manifest.jsonl", DATA / "manifest.jsonl"), ("audio", audio_dir)):
        link = data_root / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
else:
    data_root = DATA  # train_asr.py extracts audio.zip itself
print("data_root:", data_root, flush=True)

experiments = json.loads((CODE / "experiments.json").read_text(encoding="utf-8"))
n_gpu = max(torch.cuda.device_count(), 1)
print(f"gpus={n_gpu} experiments={[e['name'] for e in experiments]}", flush=True)

env = {**os.environ, "PYTHONPATH": "/tmp/pkg"}
queue = list(experiments)
running = {}  # device -> (name, process)
failed = []
tick = 0
while queue or running:
    for i in range(n_gpu):
        device = f"cuda:{i}"
        if device not in running and queue:
            exp = queue.pop(0)
            log = open(f"/kaggle/working/{exp['name']}.log", "w")
            cmd = [
                "python", str(CODE / "train_asr.py"),
                "--data-dir", str(data_root),
                "--out", f"/kaggle/working/{exp['name']}",
                "--device", device,
            ] + exp.get("args", [])
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd="/kaggle/working")
            running[device] = (exp["name"], proc)
            print("started", exp["name"], "on", device, flush=True)
    for device, (name, proc) in list(running.items()):
        code = proc.poll()
        if code is not None:
            print("finished", name, "exit code", code, flush=True)
            if code != 0:
                failed.append(name)
                print(Path(f"/kaggle/working/{name}.log").read_text(errors="replace")[-3000:], flush=True)
            del running[device]
    tick += 1
    if tick % 15 == 0:  # about every 5 minutes
        for device, (name, _) in running.items():
            lines = Path(f"/kaggle/working/{name}.log").read_text(errors="replace").splitlines()
            print(f"[progress] {name}: {lines[-1][:300] if lines else ''}", flush=True)
    time.sleep(20)

summary = {}
for exp in experiments:
    path = Path(f"/kaggle/working/{exp['name']}/history.json")
    if path.exists():
        summary[exp["name"]] = json.loads(path.read_text(encoding="utf-8"))
Path("/kaggle/working/summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
for name, history in summary.items():
    for rec in history:
        val = rec.get("val", {})
        print(
            name,
            f"epoch={rec['epoch']} loss={rec['train_loss']:.4f}",
            " ".join(f"{k}_wer={v['wer']:.4f}" for k, v in val.items()),
            flush=True,
        )
if failed:
    raise SystemExit(f"failed experiments: {failed}")
