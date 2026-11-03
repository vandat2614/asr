# ADR-0006: ASR fine-tuning strategy

**Status:** Accepted
**Date:** 2026-11-01
**Deciders:** project owner

## Context
PhoASR-small is the primary ASR model (ADR-0005). Zero-shot WER is about 0.255
(primary validation) and 0.259 (strict validation). Error analysis (worklog)
shows that 92% of errors fall on words that are frequent in train, so the model
mostly needs acoustic and domain adaptation, not vocabulary additions.
Labeled data: 1545 samples, 5036 utterances, 6.83 h. The two validation sets
(309 samples each) overlap by 58 samples, so keeping both clean leaves 985
samples (3201 utterances, about 4.3 h) for training in experiments.
Data lives in the Kaggle dataset vandat2601/zalo-lifemate-asr
(audio.zip + manifest.jsonl). Training runs in a Kaggle notebook on T4.

## Decision
1. Experiments train on utterances with `held_out == false` and are evaluated
   on both validation sets (primary and strict), which are then both clean.
2. The final model is retrained on all labeled data with the hyperparameters
   chosen in step 1 and a fixed number of epochs.
3. Training target is the raw label text (NFC only: original case, punctuation,
   digits and colloquial forms). Evaluation uses `normalize_text` (ADR-0003).
   Rationale: the downstream LLM will see ASR text and gold transcripts share
   the label style, so matching it reduces train/test mismatch.
4. Method: full fine-tuning of all weights with a low learning rate. Start
   point: AdamW, lr 1e-5, weight decay 0.01, warmup 10%, 5 epochs, effective
   batch 16, fp16 autocast with gradient scaling, seed 17, greedy decoding.
5. Evaluate after every epoch; keep the checkpoint with the lowest mean of the
   primary and strict normalized WER.
6. Code (`normalize.py`, `metrics.py`, `train_asr.py`) is shipped to Kaggle as a
   second dataset so the git repo stays the single source of truth.

## Options Considered

### Option A: Freeze most layers (conservative fine-tune)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost (VRAM, latency, time) | Low VRAM, fast |
| Expected accuracy | Another team reported a very small WER gain (0.280 to 0.276) and no downstream gain |

**Pros:** little risk of overfitting.
**Cons:** limited room to fix acoustic confusions.

### Option B: Full fine-tuning, low learning rate (chosen)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost (VRAM, latency, time) | About 4 GB for weights, gradients and Adam states, fits a T4 |
| Expected accuracy | Highest ceiling; risk of overfitting on 3201 utterances |

**Pros:** can fix acoustic and dialect confusions on action words.
**Cons:** needs the strict validation set as a guard against memorizing templates.

### Option C: LoRA adapters
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium (extra dependency, merge step for the final model) |
| Cost (VRAM, latency, time) | Lower memory than B |
| Expected accuracy | Between A and B |

**Pros:** cheap, regularizing.
**Cons:** adds moving parts; the model is only 0.24B so memory is not a constraint.

## Trade-off Analysis
The model is small enough that full fine-tuning fits on a T4, so memory
savings from LoRA buy nothing. The main risk is overfitting to templated
sentences; the strict validation set and per-epoch evaluation address it.
Option A stays as a fallback if B overfits.

## Consequences
- Easier: one training path, one evaluation path, reproducible via the repo.
- Harder: experiments use only 64% of the data; the final run must be repeated
  on all data with a fixed epoch count (no validation left for early stopping).
- Revisit: if validation WER stops improving after the first epoch (switch to
  Option A or C), or if WER improves but downstream tool-call score does not.
- WER is a proxy; a keyword-level check (nhắc, nhắn, dời, họp, báo, ví, lịch,
  chín, trưa, khoá) will be added to the report.

## Action Items
1. [ ] Write `train_asr.py` and the Kaggle code dataset.
2. [ ] Run experiment E1 (full fine-tune, lr 1e-5, 5 epochs).
3. [ ] Compare against zero-shot on both validation sets and on keywords.

## Amendment 2026-11-03: SpecAugment

Full fine-tuning without augmentation overfits (train loss 0.008 while validation WER is flat from
epoch 2). Enabling SpecAugment (time and feature masks) lowers primary WER from 0.1272 to 0.1107
with masks 0.1/0.1. The final recipe uses SpecAugment; the mask strength is tuned on validation.
