# ADR-0002: Train/validation split strategy

**Status:** Accepted
**Date:** 2026-11-01
**Deciders:** project owner

## Context
Only train.jsonl has labels (1545 samples, 5036 utterances). Public test has no
transcripts, and private test is released after the Docker image is frozen, so
all model selection must use a validation set carved out of train.
Samples can share templates (the same sentence with different slots), and
fine-tuning on one and validating on another would inflate the score.
All utterances of a sample (final request and history turns) must stay together.

## Decision
Use a grouped split (20% validation, seed 20261101, stored as sample ids in
`splits/`):
- Primary split: near-duplicate final transcripts (character-level
  SequenceMatcher ratio >= 0.9 after NFC, lowercase, punctuation removal) are
  merged into one group. Used for model selection.
- Strict split (`splits/strict/`): same procedure with threshold 0.8. Used as
  a stress test; compare model ranking against the primary split.

## Options Considered

### Option A: Random split by sample
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost (VRAM, latency, time) | None |
| Expected accuracy | Optimistic validation, template leakage |

**Pros:** trivial.
**Cons:** near-duplicate sentences land on both sides.

### Option B: Grouped split, threshold 0.9 (primary)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost (VRAM, latency, time) | About 1 minute to build |
| Expected accuracy | 98 samples (6%) in 30 multi-sample groups, largest group 10 |

**Pros:** removes most duplicates, stays close to the test distribution if test
shares templates with train.
**Cons:** same-template sentences below 0.9 can still cross the boundary.

### Option C: Grouped split, threshold 0.8 (strict)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost (VRAM, latency, time) | Same as B |
| Expected accuracy | 285 samples (18%) in 44 multi-sample groups, largest group 17 |

**Pros:** almost no template overlap, a pessimistic estimate.
**Cons:** may underestimate the real score if test overlaps train templates.

## Trade-off Analysis
We cannot measure train/test overlap because public test has no transcripts.
A single threshold would either hide leakage (0.9) or be too harsh (0.8), so we
use both and treat a large gap between them as a warning sign.

## Consequences
- Easier: leakage-aware model selection with a reproducible split.
- Harder: every experiment is reported on two validation sets.
- Revisit: if rankings differ between the two splits, or when a fine-tuned
  model is compared against the public leaderboard.
- Limitation: no speaker metadata, so the same speaker can appear on both sides.

## Action Items
1. [ ] Report WER on both splits for every candidate model.
2. [ ] Define text normalization for WER in a separate ADR.
