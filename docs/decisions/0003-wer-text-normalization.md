# ADR-0003: Text normalization for WER

**Status:** Accepted
**Date:** 2026-11-01
**Deciders:** project owner

## Context
Train labels are written as full sentences: 99.9% have capital letters and 100%
have punctuation. Numbers are mostly spelled out (about 4175 number-word tokens
vs 438 digit tokens in user utterances), but both styles occur. For example
"26 do" (54 utterances) vs "hai muoi sau do" (40), and "8 gio" (105) vs
"tam gio" (517). Money is spoken as "nghin" (125) or "ngan" (20), and the
D.DDD format never appears in user utterances (only in assistant history).
A raw comparison would report errors for differences that do not change which
tool the agent calls.

## Decision
Primary WER is computed after `normalize_text` (src/asr/normalize.py), applied
to both reference and hypothesis:
1. Unicode NFC, lowercase.
2. Digits to Vietnamese words: times (8h30, 8:30), thousands separators,
   decimals, percent. Numbers with a leading zero or more than 12 digits are
   read digit by digit (phone numbers).
3. Merge spelling variants: ngan -> nghin, lam -> nam, mot (the "mot" with
   tone mark) -> mot, le -> linh, "muoi tu" -> "muoi bon". The word "tu" is
   kept elsewhere (for example "thu tu").
4. Punctuation replaced by spaces, whitespace collapsed.
A secondary "raw" WER uses `normalize_basic` (steps 1 and 4 only). The gap
between the two shows how much error comes from number formatting.
The raw ASR text is passed unchanged to the LLM.

## Options Considered

### Option A: Raw comparison
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost (VRAM, latency, time) | None |
| Expected accuracy | Inflated WER from casing, punctuation and number style |

**Pros:** no assumptions.
**Cons:** hides real progress behind formatting noise.

### Option B: Lowercase and strip punctuation only
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost (VRAM, latency, time) | None |
| Expected accuracy | Still penalizes "26" vs "hai muoi sau" |

**Pros:** simple and standard.
**Cons:** number style differences remain (digits are 9.6% of utterances).

### Option C: Option B plus digits to words and variant merging (chosen)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost (VRAM, latency, time) | Small module with unit tests |
| Expected accuracy | Removes formatting noise, keeps real recognition errors |

**Pros:** about 90% of labels are already in word form, so the mapping is
mostly one-directional and deterministic.
**Cons:** hand-written reading rules can be wrong in rare cases.

### Option D: Convert words to digits
| Dimension | Assessment |
|-----------|------------|
| Complexity | High |
| Cost (VRAM, latency, time) | Parsing spoken numbers is ambiguous |
| Expected accuracy | Error-prone |

**Pros:** shorter token sequences.
**Cons:** harder than the reverse direction and most labels are words.

## Trade-off Analysis
WER here is a proxy for downstream tool-call accuracy, so it should ignore
differences the agent does not care about. Option C does that at the cost of
maintaining a small normalizer. Reporting the raw WER next to it keeps the
choice auditable.

## Consequences
- Easier: stable, comparable WER across models with different number styles.
- Harder: normalizer rules must be maintained and tested.
- Not covered: slot-level correctness (names, addresses, device names). WER
  can hide a wrong name inside an otherwise correct sentence, so a slot-level
  check will be added later.
- Revisit: after running the normalizer over all train transcripts and after
  seeing real model output.

## Action Items
1. [ ] Run normalize_text over all train transcripts and review leftovers
   (digits, unusual symbols).
2. [ ] Add a WER script that reports both primary and raw WER.

## Amendment 2026-11-01: optional "mươi" between tens and units
Error analysis showed that speakers often use the short spoken form ("bốn lăm"
for 45, "chín lăm" for RON95) while labels use digits, which the normalizer
expanded to the full form. This produced artificial deletions of "mươi".
The normalizer now drops "mươi" when it is followed by a digit word, on both
the reference and the hypothesis. Bare tens ("ba mươi phần trăm") keep it.
Effect: about 0.1 WER point lower for every model, ranking unchanged.
