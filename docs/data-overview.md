# Data overview

Our own summary of the LifeMate data package. The data itself is under
`data/` and is git-ignored: the organisers forbid redistribution.

## Layout (after extraction under `data/raw/train+public_test/`)

- `train/` and `public_test/`, each with `audio/`, `<split>.jsonl`,
  `<split>_tools_catalog.json`, `<split>_environment/` and `zaic_tools/`.
- `private_test/` is released only after the final Docker image is frozen.
- Audio naming: `audio/<id>/<id>.wav` is the final request (the scored one);
  `<id>_u1.wav`, `<id>_u2.wav`, ... are earlier user turns in the history.
- Rule: never read or parse `*_environment/` (including `state.json`).

## Statistics (from `scripts/data_stats.py`, 2026-11-01)

| | train | public_test |
|---|---|---|
| Samples | 1545 | 634 |
| Samples with history | 1018 (66%) | 419 (66%) |
| History user turns | 3491 | 1299 |
| Wav files (= utterances) | 5036 | 1933 |
| Total audio | 6.83 h | 2.76 h |
| Mean / p50 / p95 / max (s) | 4.88 / 4.11 / 9.86 / 22.33 | 5.14 / 4.27 / 10.77 / 20.24 |

- Every wav is readable and every final audio path exists.
- Train has a transcript (`audio_text`) for all 5036 utterances
  (1545 final + 3491 history). Public test has no transcripts.
- All utterances are shorter than 30 s, so no chunking is needed.

## Implications for the ASR module

- Inference must transcribe about 3 utterances per sample (history audio has
  no text at test time), inside the 30 s per-sample budget shared with the LLM.
- About 6.8 h of labeled speech: enough for LoRA domain adaptation, not for
  training from scratch.
- Validation split must keep all utterances of one sample on the same side and
  group near-duplicate transcripts, to avoid leakage.
- No speaker or region metadata is available, so we cannot stratify by dialect.
