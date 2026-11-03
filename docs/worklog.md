# Worklog

## 2026-11-01
- Created repo, docs skeleton and ADR template.
- ADR-0001: development compute (Kaggle T4 x2).
- Downloaded and extracted the data package into data/raw/ (1.2 GB).
- Removed __MACOSX junk. Wrote scripts/data_stats.py.
- Data check: train 1545 samples / 5036 utterances / 6.83 h; public_test 634 samples / 1933 utterances / 2.76 h. No missing or unreadable audio.
- Wrote docs/data-overview.md.
- Wrote scripts/make_split.py (grouped split by near-duplicate final transcripts).
- Primary split (threshold 0.9): 1236 train / 309 val samples, 1477 groups, 98 samples in multi-sample groups.
- Strict split (threshold 0.8): 1236 train / 309 val samples, 1304 groups, 285 samples in multi-sample groups.
- ADR-0002: use 0.9 as primary and 0.8 as a stress test.
- Wrote scripts/transcript_stats.py and scripts/number_style_stats.py (aggregate counts only, no transcript text).
- Findings: labels mostly spell numbers out; times appear as "D gio" or word form (no 8:30 form); money uses nghin/ngan; D.DDD only in assistant text.
- Wrote src/asr/normalize.py with unit tests (10 passing).
- ADR-0003: primary WER on normalized text, secondary raw WER.
- Kaggle notebook serves ASR via FastAPI + cloudflared tunnel; VM client in src/asr/remote.py (ADR-0004). Metrics in src/asr/metrics.py. 14 unit tests pass.
- Smoke test, vinai/PhoWhisper-small, 20 validation utterances (preliminary, tiny sample): WER primary 0.286, raw 0.346, CER 0.154, 0.26 s GPU per utterance on T4.
- Observed errors concentrate on time words (mai, gio, tieng, doi), person names and some opening phrases. Labels contain digits in some utterances while the model writes words, which the normalizer handles.
- Wrote scripts/transcribe_remote.py (resumable, saves raw hypotheses to outputs/asr/<model>/) and scripts/score_asr.py.
- Baseline vinai/PhoWhisper-small on full validation: primary WER 0.3259 / raw 0.3374 / CER 0.1812 (989 utts); strict WER 0.3260 / raw 0.3440 / CER 0.1810 (1044 utts); 0.094 s GPU per utterance on T4.
- Number-style normalization is worth about 1-2 WER points on the full set. Final-request and history utterances are equally hard.
- Zero-shot benchmark on full validation (WER primary / strict / final-only primary / final-only strict):
  - vinai/PhoWhisper-large: 0.2419 / 0.2578 / 0.2337 / 0.2327, 0.348 s GPU per utterance
  - Qualcomm-AI-Research/PhoASR-whisper-small: 0.2557 / 0.2602 / 0.2633 / 0.2548, 0.087 s
  - vinai/PhoWhisper-medium: 0.2634 / 0.2726 / 0.2550 / 0.2595, 0.219 s
  - vinai/PhoWhisper-small: 0.3259 / 0.3260 / 0.3241 / 0.3165, 0.094 s
- PhoASR-small is the best size/accuracy trade-off; large leads on final requests but there are only about 309 of them per split, so the gap is not proven.
- Finalists: PhoASR-small and PhoWhisper-large. PhoWhisper-small dropped; medium nearly dominated.
- ADR-0005: PhoASR-whisper-small chosen as primary ASR (faster fine-tuning, leaves parameter budget for the LLM). PhoWhisper-large kept as accuracy reference only.
- Error analysis of PhoASR-small on the primary validation split (scripts/error_analysis.py): 91.7% of reference-word errors fall on words seen more than 5 times in train; unseen or rare words account for about 8%. Error rate: number words 17.1%, time-related words 16.2%.
- Main confusion groups: colloquial forms rewritten as standard Vietnamese (hông->không, nha->nhé), near-homophones on action words (nhắc->nhất, dời->giờ, họp->học, chín->chính), dropped fillers (ờ, ừ).
- scripts/show_examples.py showed that "mươi" deletions are a scoring artifact (speakers say "bốn lăm", labels use digits), while "khoá" deletions are real recognition errors. Some utterances are nearly unrecognized (for example "xăng RON95", "phòng khách" heard as "phong cách").
- Normalizer: "mươi" is now optional between tens and units. 21 tests pass. WER drops about 0.1 point for all models; ranking unchanged. PhoASR-small now 0.2547 (primary) / 0.2594 (strict).
- Wrote scripts/build_asr_dataset.py: audio.zip (614 MB) + manifest.jsonl with split flags. Uploaded to Kaggle as vandat2601/zalo-lifemate-asr (private).
- Held-out design: primary val and strict val overlap by 58 samples, so 560 samples are held out and 985 samples (3201 utterances, about 4.3 h) remain for experiments; the final model is retrained on all 1545 samples.
- ADR-0006: full fine-tuning at a low learning rate on raw label text, per-epoch evaluation on both validation sets.
- Wrote scripts/train_asr.py (full fine-tune, per-epoch evaluation on both validation sets), scripts/kaggle_run.py (runs experiments from a JSON config inside a Kaggle script kernel, one per GPU) and scripts/run_kaggle.sh (push code and config, start the kernel from the VM).
- Smoke test through the Kaggle kernel worked end to end; results match the interactive notebook run (selection WER identical, loss differs in the 4th decimal).
- Next: E1 (full fine-tune) and E2 (frozen encoder) with configs/kaggle_e1_e2.json.

## 2026-11-02 - E1/E2 fine-tuning on Kaggle (experiment mode, train on clean split only)

Setup: PhoASR-small, lr 1e-5, 5 epochs, fp16, greedy decoding. Eval on the union of both validation sets.
Zero-shot baseline: primary WER 0.2547, strict WER 0.2594, primary CER 0.1451.

| Exp | Best epoch | Primary WER | Strict WER | Primary CER | Wall time |
|---|---|---|---|---|---|
| e1 full fine-tune | 5 (epoch 2 within noise) | 0.1288 | 0.1330 | 0.0765 | ~44 min |
| e2 frozen encoder | 3 | 0.1710 | 0.1782 | 0.1020 | ~27 min |

Findings:
- E1 halves WER vs zero-shot (-12.6 points absolute). Primary and strict agree, so the split does not look leaky.
- Frozen encoder loses ~4 points: acoustic adaptation matters most.
- E1 val WER plateaus from epoch 2 while train loss keeps falling; epochs 3-5 add nothing measurable.
- Caveat: validation shares the scenario distribution with train; test WER is likely higher.

Next: per-category error analysis of e1/best, then `--final` run on all data (about 4 epochs).

## 2026-11-02 - Inference check and error analysis of e1/best

Inference with the e1/best checkpoint (Kaggle kernel `zalo-asr-infer`, greedy, fp16) reproduces the
training-time numbers (primary 0.1286 vs 0.1288, strict 0.1330 vs 0.1330).

Error analysis (e1, primary):
- Frequent words carry 86% of the errors; rare and unseen words only ~14%.
- Time words (6.5%) and number words (6.0%) are below the average error rate.
- Many top substitutions are label-style variants, not recognition errors: không/hông, nghe/nha/nhé,
  giùm/dùm, tui/tôi, and old/new tone placement.
- Meaning-changing confusions worth tracking: dời->giờ (7), học->họp, hạn->hẹn, tối->tới.

Normalizer amendment (ADR-0003): unify old/new tone placement on open syllables.
New reference numbers: zero-shot PhoASR-small primary 0.2542 / strict 0.2589;
e1 primary 0.1272 / strict 0.1316.

Next: keyword-level metric for tool-call relevant words; then final run on all data (about 4 epochs).

## 2026-11-03 - E3/E4: SpecAugment (experiment mode, new normalizer)

Same recipe as e1 plus SpecAugment (time/feature mask probability). Reference e1 under the new
normalizer: primary 0.1272, strict 0.1316, CER 0.0758.

| Exp | Masks (time/feature) | Primary WER | Strict WER | Primary CER | Final train loss |
|---|---|---|---|---|---|
| e3 | 0.05 / 0.05 | 0.1131 (ep5) | 0.1175 | 0.0690 | 0.098 |
| e4 | 0.10 / 0.10 | 0.1107 (ep4=ep5) | 0.1126 | 0.0672 | 0.166 |

Findings:
- SpecAugment removes most of the overfitting: -1.65 points primary, -1.9 points strict vs e1.
- Stronger masking helps, with diminishing returns (e3 -> e4: -0.24 points primary).
- Selection was done on the same validation sets used for reporting, so numbers are slightly optimistic.

Next: `--final` on all data with the e4 recipe (final_e4) and e5 (masks 0.15) in parallel.

## 2026-11-03 - E5, final run, offline backend and decoding variants

- e5 (SpecAugment 0.15/0.15): primary 0.1074, strict 0.1126 at epoch 5. Within noise of e4
  (0.1107 / 0.1126), so the e4 recipe is kept. Further mask tuning is not worth the GPU time.
- final_e4 (e4 recipe, all 1545 samples, 5 epochs, last epoch kept): trained, train loss 0.140.
  It has no held-out validation; its quality is inferred from e4.
- `src/asr/local.py`: offline backend (`LocalASR.transcribe(paths) -> list[str]`). CPU smoke test on
  8 utterances loads from the checkpoint directory without network, 7/8 identical to GPU output.
- Decoding variants of e4-best on the validation union (primary / strict WER, final-only number
  accuracy, final-only name recall):

| Variant | Primary | Strict | Numbers right | Name recall |
|---|---|---|---|---|
| greedy fp16 | 0.1108 | 0.1128 | 0.896 | 0.812 |
| greedy fp32 | 0.1106 | 0.1127 | 0.896 | 0.812 |
| beam5 fp16 | 0.1030 | 0.1052 | 0.911 | 0.824 |
| beam5 fp32 | 0.1030 | 0.1052 | 0.908 | 0.826 |

Decision: beam search with 5 beams, fp16. Precision makes no measurable difference.

## 2026-11-04 - ASR module v0.1 (interim)

Packaged `models/asr-final` (final_e4) with `LocalASR` (beam 5, batch 1) and `scripts/transcribe_local.py`.
CPU smoke test on 8 utterances: the final model matches the labels on 5/8 after normalization and
fixes all three utterances that e4-best got wrong. Known gaps are listed in `docs/asr-module.md`.
The batch-size determinism check was deliberately postponed.
