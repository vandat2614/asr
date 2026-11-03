# ADR-0007: Offline ASR backend and decoding settings

## Status
Accepted (2026-11-03)

## Context
The final solution must run offline, deterministically, and within 30 s per sample on one A30.
Until now ASR ran through a development-only remote service. We need a production backend with the
same `transcribe(paths) -> list[str]` interface, and a decoding configuration chosen with data.

## Decision
- `src/asr/local.py` (`LocalASR`) loads a fine-tuned Whisper checkpoint from a local directory,
  with `HF_HUB_OFFLINE=1`, and never uses the network.
- Decoding: beam search, 5 beams, fp16 on GPU, language `vi`, task `transcribe`, no sampling.
- The checkpoint is trained with SpecAugment (time/feature masks 0.1) on all labeled data.

## Alternatives considered
- Greedy decoding: 0.75 points worse WER, slightly worse numbers and names, marginally faster.
- fp32: no measurable gain over fp16 and 2x slower with beam search.
- Larger Whisper models: rejected in ADR-0005 (parameter budget).

## Consequences
- Beam search costs about 1.5x the time of greedy decoding; still far below the 30 s limit.
- The decoding choice was made on the same validation sets used to select the recipe, so the
  reported numbers are slightly optimistic.
- The ASR text is not yet verified end to end against tool-call accuracy (no agent in the loop).
