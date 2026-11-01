# ADR-0005: ASR model selection

**Status:** Accepted
**Date:** 2026-11-01
**Deciders:** project owner

## Context
Zero-shot results on the full validation sets (WER primary / strict):
PhoWhisper-large 0.242 / 0.258, PhoASR-small 0.256 / 0.260,
PhoWhisper-medium 0.263 / 0.273, PhoWhisper-small 0.326 / 0.326.
On final requests only (about 309 utterances per split) large leads by
roughly 2-3 points, but that sample is small and the gap is not proven.
The whole system must stay under 6B parameters (LLM under 5B), shared with
the LLM agent.

## Decision
Use Qualcomm-AI-Research/PhoASR-whisper-small (about 0.24B parameters) as the
primary ASR model and the base for fine-tuning. Keep PhoWhisper-large only as
an accuracy reference.

## Options Considered

### Option A: PhoASR-small (chosen)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost (VRAM, latency, time) | 0.087 s GPU per utterance on T4, quick to fine-tune |
| Expected accuracy | WER 0.256 / 0.260 zero-shot |

**Pros:** small, fast to iterate, leaves most of the parameter budget to the LLM.
**Cons:** about 1.4 points behind large on the primary split.

### Option B: PhoWhisper-large
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost (VRAM, latency, time) | 0.348 s GPU per utterance, about 1.55B parameters, slow to fine-tune on T4 |
| Expected accuracy | WER 0.242 / 0.258 zero-shot |

**Pros:** best zero-shot accuracy, especially on final requests.
**Cons:** consumes about a quarter of the parameter budget, harder to fine-tune.

### Option C: PhoWhisper-medium
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost (VRAM, latency, time) | 0.219 s GPU per utterance, about 0.77B parameters |
| Expected accuracy | WER 0.263 / 0.273 zero-shot |

**Pros:** none over A on overall WER.
**Cons:** larger and slower than A with a worse overall WER.

## Trade-off Analysis
The accuracy gap between A and B is small and partly within noise, while B costs
over 1.3B parameters and much slower experiments. Faster iteration and a larger
budget for the LLM are worth more than one or two WER points, because the final
score depends on tool calls, not on WER.

## Consequences
- Easier: quick fine-tuning cycles on a single T4, more room for the LLM.
- Harder: we give up large's lead on final requests unless fine-tuning closes it.
- Revisit: if, after fine-tuning, downstream scores show ASR is the bottleneck
  and the LLM leaves at least 1.5B of budget free, re-test large.

## Action Items
1. [ ] Analyze PhoASR-small errors by category (numbers, time words, rare words).
2. [ ] Decide between fine-tuning and hotwords based on that analysis.
