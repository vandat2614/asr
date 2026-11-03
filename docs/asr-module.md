# ASR module (v0.1, interim)

Offline Vietnamese ASR for the LifeMate pipeline.

## Usage
```python
from asr.local import LocalASR
asr = LocalASR("models/asr-final")          # beam 5, batch 1, fp16 on GPU / fp32 on CPU
texts = asr.transcribe(["a.wav", "b.wav"])  # list[str], same order as the input
```
CLI: `python3 scripts/transcribe_local.py a.wav b.wav`

## What is inside
- `models/asr-final`: PhoASR-whisper-small (0.24B parameters) fully fine-tuned with SpecAugment
  (time/feature masks 0.1) on all labeled data (train + former validation), 5 epochs.
  The directory is git-ignored; it is about 967 MB and must be copied into the Docker image.
- `src/asr/local.py`: backend. Network access is disabled (`HF_HUB_OFFLINE=1`).
  Decoding is beam search with 5 beams, no sampling, fixed seed.

## Measured quality
Numbers come from the held-out recipe run (e4-best, beam 5), because `models/asr-final` has no
independent validation set (see ADR-0007, worklog):
- WER 0.103 (primary) / 0.105 (strict) vs 0.254 / 0.259 zero-shot.
- Final utterances: 91% have every number right, name-token recall 0.82.

## Known limitations and next steps
- Not yet verified inside the target Docker image on an A30; end-to-end time per sample is unmeasured.
- Batch-size dependence of fp16 output is not measured; the default batch size 1 avoids the issue.
  A same-input-same-output check on GPU is still to do (scripts/infer_determinism_driver.py).
- No LLM agent in the loop yet, so tool-call accuracy is unknown.
- Rare or unseen names (foreign names, brands, short names like Bống, Bi) are the weakest slot.
  Plan: on the agent side, retry with `contacts.search()` (whole address book) and fuzzy-match.
- Numbers are exact-match critical (they enter `state_hash`); about 9% of final utterances still
  contain a wrong number.
- Possible gains: more epochs with stronger masks (e5 was still improving), more data augmentation.
