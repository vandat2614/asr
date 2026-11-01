# ADR-0004: Remote ASR service for development

**Status:** Accepted
**Date:** 2026-11-01
**Deciders:** project owner

## Context
The VM that holds the repo and the data has no GPU. Kaggle offers T4 x2 for
free. Pushing code and data to a Kaggle notebook for every experiment is slow
and keeps two copies of everything in sync.
The competition requires fully offline, local inference in the final Docker
image (no web service, no external API), so any remote setup is development
only.

## Decision
Run the ASR model as a small FastAPI service in a Kaggle notebook, exposed by
a cloudflared quick tunnel and protected by a token header. The VM calls it
through `asr.remote.RemoteASR`.
`RemoteASR.transcribe(paths) -> list[str]` is the interface; a future local
backend must implement the same method so the final solution can swap
backends without touching the pipeline.
Decoding is deterministic (greedy, num_beams=1, no sampling, fixed seed).

## Options Considered

### Option A: Push code and data to a Kaggle notebook
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost (VRAM, latency, time) | Free; re-upload per change |
| Expected accuracy | Same as other options |

**Pros:** no tunnel, no network in the loop.
**Cons:** slow iteration, two copies of code and data.

### Option B: Remote service on Kaggle via tunnel (chosen)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost (VRAM, latency, time) | Free; about 0.26 s GPU per utterance (whisper-small, T4) |
| Expected accuracy | Same as other options |

**Pros:** code and data stay on the VM, switching model means restarting one
notebook cell, benchmark logic stays in the repo.
**Cons:** tunnel URL changes on every restart, Kaggle session and quota limits,
network time is mixed into wall-clock time.

### Option C: Rent a GPU machine (vast.ai or similar)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost (VRAM, latency, time) | Paid, no session limits |
| Expected accuracy | Same as other options |

**Pros:** stable, full control, no tunnel.
**Cons:** costs money, needs its own setup.

## Trade-off Analysis
Option B gives the fastest iteration at no cost. Its weaknesses (unstable
URL, quotas) only affect convenience, not results, and Option C stays
available as a fallback.

## Consequences
- Easier: swapping models and running benchmarks from the VM.
- Harder: every Kaggle restart needs a new URL in `.env`.
- Latency from this service is server-side GPU time on a T4 (reported by the
  service). Wall-clock time includes the network and must not be used as the
  latency estimate for the final solution.
- Fp16 batch results can differ slightly with batch composition; the final
  local backend must fix batch size and ordering for reproducibility.
- Revisit if Kaggle quota becomes a bottleneck (switch to Option C).

## Action Items
1. [ ] Implement the local backend with the same interface.
2. [ ] Write the benchmark script over the full validation sets.
