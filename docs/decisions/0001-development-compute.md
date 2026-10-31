# ADR-0001: Development compute

**Status:** Accepted
**Date:** 2026-11-01
**Deciders:** project owner

## Context
We need to benchmark and fine-tune Vietnamese ASR models. The scoring machine is a single NVIDIA A30 (24 GB), so anything we develop must fit that target.

## Decision
Use Kaggle GPU T4 x2 for development. Rent a GPU on vast.ai only when needed.

## Options Considered

### Option A: Kaggle T4 x2
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost (VRAM, latency, time) | Free, 16 GB per GPU, fp16 only (no bf16), weekly quota |
| Expected accuracy | Enough for small/medium ASR and LoRA fine-tuning |

**Pros:** free, no setup, two GPUs for parallel runs.
**Cons:** session and quota limits, slower than A30.

### Option B: Rented GPU on vast.ai
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost (VRAM, latency, time) | Paid, flexible VRAM and session length |
| Expected accuracy | Allows larger models and full fine-tuning |

**Pros:** no quota, larger VRAM.
**Cons:** costs money, needs environment setup.

## Trade-off Analysis
Kaggle is enough for the current phase. Paying for a rented GPU is only justified when a model or experiment does not fit Kaggle.

## Consequences
- Latency measured on T4 is a conservative estimate compared with the A30.
- Models must run in fp16.
- Revisit if we need a model larger than about 1.5B parameters or many parallel runs.

## Action Items
1. [ ] Re-measure latency on an A30-class GPU before freezing the final solution.
