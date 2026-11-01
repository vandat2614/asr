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
