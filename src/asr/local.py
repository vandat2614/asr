"""Offline ASR backend: a fine-tuned Whisper checkpoint on local disk.

Same interface as RemoteASR: transcribe(paths) -> list[str]. Decoding is deterministic
(greedy or beam search, never sampling) and never touches the network.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

SR = 16000
MAX_NEW_TOKENS = 200


def _load_audio(path):
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import torchaudio  # only needed for non-16 kHz input

        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    return np.ascontiguousarray(wav)


class LocalASR:
    def __init__(self, ckpt_dir, device=None, num_beams=5, batch_size=1, precision="auto"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if precision == "auto":
            precision = "fp16" if self.device.startswith("cuda") else "fp32"
        self.dtype = {"fp16": torch.float16, "fp32": torch.float32}[precision]
        self.num_beams = num_beams
        self.batch_size = batch_size
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.processor = WhisperProcessor.from_pretrained(ckpt_dir)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(ckpt_dir, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def transcribe(self, paths):
        order = sorted(range(len(paths)), key=lambda i: os.path.getsize(paths[i]))
        out = [None] * len(paths)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            feats = self.processor.feature_extractor(
                [_load_audio(paths[i]) for i in idx], sampling_rate=SR, return_tensors="pt"
            ).input_features.to(self.device, dtype=self.dtype)
            ids = self.model.generate(
                input_features=feats,
                language="vi",
                task="transcribe",
                num_beams=self.num_beams,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            for i, text in zip(idx, self.processor.batch_decode(ids, skip_special_tokens=True)):
                out[i] = text.strip()
        return out
